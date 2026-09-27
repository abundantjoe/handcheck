"""tifxyz surface I/O and basic grid geometry.

On-disk format (as written and read by the official
ScrollPrize/villa code, https://github.com/ScrollPrize/villa):

* directory containing ``x.tif``, ``y.tif``, ``z.tif`` and ``meta.json``
  (optional ``mask.tif`` and extra channels such as ``generations.tif``);
* each coordinate TIFF is a single-page, single-sample float32 image of shape
  (rows, cols) = (H, W); row index = grid v, column index = grid u;
  the value is the voxel coordinate of that grid vertex in the volume
  (x = fastest volume axis, z = slice index);
* invalid vertices are (-1, -1, -1); the official loader additionally
  invalidates every vertex with z <= 0 and every vertex whose ``mask.tif``
  channel-0 value is below 255;
* ``meta.json``: ``{"bbox": [[xmin,ymin,zmin],[xmax,ymax,zmax]],
  "format": "tifxyz", "scale": [scale_u, scale_v], "type": "seg",
  "uuid": "..."}`` plus optional extra keys; ``scale`` is grid vertices per
  voxel, i.e. the nominal grid step is ``1/scale`` voxels (0.05 -> 20 vox).

Grid convention used throughout ``handcheck``: ``points`` is an
(H, W, 3) float array, ``points[v, u] = (x, y, z)``; u = column, v = row.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

SENTINEL = -1.0
COORD_FILES = ("x.tif", "y.tif", "z.tif")
REQUIRED_META_KEYS = ("bbox", "format", "scale", "type", "uuid")
# official loader keeps a mask pixel iff channel 0 >= 255 (QuadSurface.cpp)
MASK_RETAIN_THRESHOLD = 255.0


@dataclass
class TifxyzMesh:
    """In-memory tifxyz surface.

    points : (H, W, 3) float32, ``points[v, u] = (x, y, z)`` in voxels,
             invalid vertices exactly (-1, -1, -1).
    scale  : (scale_u, scale_v) exactly as stored in meta.json
             (``[x_scale, y_scale]`` = [columns, rows]); grid step = 1/scale vox.
    uuid   : segment id (meta.json "uuid").
    meta   : any other meta.json keys (preserved on write, never override the
             required keys).
    """

    points: np.ndarray
    scale: tuple[float, float] = (0.05, 0.05)
    uuid: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return valid_mask(self.points)

    @property
    def step_vox(self) -> tuple[float, float]:
        """Nominal grid step (u, v) in voxels = 1/scale."""
        return (1.0 / float(self.scale[0]), 1.0 / float(self.scale[1]))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.points.shape[:2])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# validity helpers
# ---------------------------------------------------------------------------

def valid_mask(points: np.ndarray) -> np.ndarray:
    """True where a grid vertex is valid.

    Mirrors the official ``isValidPointSample``: invalid if ANY component is
    exactly -1 or any component is non-finite.
    """
    p = np.asarray(points)
    if p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError(f"points must be (H, W, 3), got {p.shape}")
    return np.all(np.isfinite(p), axis=-1) & ~np.any(p == SENTINEL, axis=-1)


def set_invalid(points: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Return a float32 copy with ``invalid`` vertices set to the sentinel."""
    out = np.array(points, dtype=np.float32, copy=True)
    out[np.asarray(invalid, dtype=bool)] = SENTINEL
    return out


def canonicalize(points: np.ndarray) -> np.ndarray:
    """float32 copy in which every invalid vertex is exactly (-1, -1, -1)."""
    return set_invalid(points, ~valid_mask(points))


def quad_valid_mask(points: np.ndarray) -> np.ndarray:
    """(H-1, W-1) mask: all four corners of the quad are valid."""
    v = valid_mask(points)
    return v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]


def bbox(points: np.ndarray) -> list[list[float]]:
    """[[xmin, ymin, zmin], [xmax, ymax, zmax]] over valid vertices.

    With no valid vertex the official code yields the -1 marker, reproduced
    here.
    """
    v = valid_mask(points)
    if not v.any():
        return [[SENTINEL] * 3, [SENTINEL] * 3]
    p = np.asarray(points, dtype=np.float32)[v]
    return [[float(c) for c in p.min(axis=0)], [float(c) for c in p.max(axis=0)]]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_band(path: Path) -> np.ndarray:
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) < 1:
            raise ValueError(f"empty TIFF: {path}")
        arr = tf.pages[0].asarray()
    if arr.ndim != 2:
        raise ValueError(f"expected single-sample 2D TIFF in {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _apply_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Official mask semantics: channel 0 < 255 invalidates; the mask may be
    an integer multiple of the grid size, and a grid vertex is invalidated if
    any of its mask pixels is below threshold."""
    m = np.asarray(mask)
    if m.ndim == 3:  # contiguous multi-sample: channel 0 is validity
        m = m[..., 0]
    H, W = points.shape[:2]
    mh, mw = m.shape
    if mh % H or mw % W:
        return points  # official loader ignores a mask with non-integer ratio
    fy, fx = mh // H, mw // W
    keep = (m.astype(np.float64) >= MASK_RETAIN_THRESHOLD)
    keep = keep.reshape(H, fy, W, fx).all(axis=(1, 3))
    return set_invalid(points, ~keep)


def read_tifxyz(path: str | os.PathLike, *, apply_mask: bool = True) -> TifxyzMesh:
    """Read a tifxyz directory the way the official C++ loader does.

    * any TIFF sample type is converted to float32;
    * vertices with z <= 0 become (-1, -1, -1) (official loader rule);
    * non-finite / sentinel vertices are canonicalised to (-1, -1, -1);
    * if ``apply_mask`` and ``mask.tif`` exists, pixels < 255 invalidate.
    """
    d = Path(path)
    for name in (*COORD_FILES, "meta.json"):
        if not (d / name).exists():
            raise FileNotFoundError(f"missing {name} in {d}")
    with open(d / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "scale" not in meta or len(meta["scale"]) < 2:
        raise ValueError(f"meta.json in {d} lacks a 2-element 'scale'")
    x, y, z = (_read_band(d / n) for n in COORD_FILES)
    if not (x.shape == y.shape == z.shape):
        raise ValueError(f"band size mismatch in {d}: {x.shape} {y.shape} {z.shape}")
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    invalid = ~valid_mask(pts) | (pts[..., 2] <= 0.0)
    pts = set_invalid(pts, invalid)
    if apply_mask and (d / "mask.tif").exists():
        pts = _apply_mask(pts, tifffile.imread(str(d / "mask.tif")))
    scale = (float(meta["scale"][0]), float(meta["scale"][1]))
    extra = {k: v for k, v in meta.items() if k not in REQUIRED_META_KEYS}
    return TifxyzMesh(points=pts, scale=scale, uuid=str(meta.get("uuid", "")), meta=extra)


def _write_band(path: Path, band: np.ndarray, dpi: float | None) -> None:
    band = np.ascontiguousarray(band, dtype="<f4")
    kwargs: dict[str, Any] = dict(
        photometric="minisblack",
        compression=None,
        rowsperstrip=band.shape[0],  # official: one strip, untiled, uncompressed
        metadata=None,  # no tifffile JSON ImageDescription
    )
    if dpi is not None and dpi > 0:
        kwargs["resolution"] = (float(dpi), float(dpi))
        kwargs["resolutionunit"] = "INCH"
    tifffile.imwrite(str(path), band, **kwargs)


def write_tifxyz(
    path: str | os.PathLike,
    mesh: TifxyzMesh,
    *,
    overwrite: bool = False,
    voxel_size_um: float | None = None,
    write_mask: bool = False,
    preserve_aux: bool = True,
) -> Path:
    """Write ``mesh`` as a tifxyz directory compatible with the official tools.

    x/y/z.tif: float32, single strip, uncompressed, minisblack; invalid
    vertices written as -1 in all three bands. meta.json: bbox, format,
    scale, type, uuid (+ ``mesh.meta`` extras), indent 4. If
    ``voxel_size_um`` is given the TIFF resolution tag is set to
    25400/voxel_size_um dpi like ``voxelSizeToDpi``. ``write_mask`` adds a
    uint8 mask.tif (255 valid / 0 invalid). The write goes to a temp dir
    first and is then renamed into place. On overwrite with
    ``preserve_aux`` (official behaviour), regular files of the old directory
    that were not rewritten (extra channels such as generations.tif, and a
    stale mask.tif) are carried forward.
    """
    d = Path(path)
    if not mesh.uuid:
        raise ValueError("mesh.uuid must be non-empty (official save requires it)")
    if d.exists() and not overwrite:
        raise FileExistsError(f"{d} exists (pass overwrite=True)")
    pts = canonicalize(mesh.points)
    if pts.shape[0] < 2 or pts.shape[1] < 2:
        raise ValueError("grid must be at least 2x2")
    d.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp_tifxyz_", dir=str(d.parent)))
    try:
        dpi = (25400.0 / voxel_size_um) if voxel_size_um else None
        for i, name in enumerate(COORD_FILES):
            _write_band(tmp / name, pts[..., i], dpi)
        meta: dict[str, Any] = {k: v for k, v in mesh.meta.items() if k not in REQUIRED_META_KEYS}
        meta.update(
            bbox=bbox(pts),
            format="tifxyz",
            scale=[float(mesh.scale[0]), float(mesh.scale[1])],
            type="seg",
            uuid=str(mesh.uuid),
        )
        with open(tmp / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, sort_keys=True)
            f.write("\n")
        if write_mask:
            m = (valid_mask(pts).astype(np.uint8) * 255)
            tifffile.imwrite(str(tmp / "mask.tif"), m, photometric="minisblack", metadata=None)
        if d.exists():
            if preserve_aux:
                for f in d.iterdir():
                    if f.is_file() and not (tmp / f.name).exists():
                        shutil.copy2(f, tmp / f.name)
            shutil.rmtree(d)
        os.replace(tmp, d)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return d


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def densify(points: np.ndarray, k: int = 5) -> np.ndarray:
    """Bilinear subdivision of every valid quad into k x k sub-quads.

    Returns a ((H-1)*k+1, (W-1)*k+1, 3) float32 grid. A dense vertex is valid
    iff it lies in (or on the border of) at least one valid quad; samples on
    shared quad edges depend only on the edge endpoints and are therefore
    identical whichever quad writes them.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    p = np.asarray(points, dtype=np.float64)
    H, W = p.shape[:2]
    qv = quad_valid_mask(p)
    out = np.full(((H - 1) * k + 1, (W - 1) * k + 1, 3), SENTINEL, dtype=np.float32)
    qj, qi = np.nonzero(qv)
    if qj.size == 0:
        return out
    p00 = p[qj, qi]
    p01 = p[qj, qi + 1]  # +u
    p10 = p[qj + 1, qi]  # +v
    p11 = p[qj + 1, qi + 1]
    for a in range(k + 1):  # v offset
        t = a / k
        for b in range(k + 1):  # u offset
            s = b / k
            val = (1 - s) * (1 - t) * p00 + s * (1 - t) * p01 + (1 - s) * t * p10 + s * t * p11
            out[qj * k + a, qi * k + b] = val
    return out


def densify_mesh(mesh: TifxyzMesh, k: int = 5) -> TifxyzMesh:
    """Densified copy of a mesh; the scale is multiplied by k."""
    return TifxyzMesh(
        points=densify(mesh.points, k),
        scale=(mesh.scale[0] * k, mesh.scale[1] * k),
        uuid=mesh.uuid,
        meta=dict(mesh.meta),
    )


def grid_derivatives(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dp/du and dp/dv per vertex (per grid index, not per voxel).

    Central difference where both neighbours are valid, one-sided where only
    one is, NaN otherwise (and at invalid vertices).
    """
    p = np.asarray(points, dtype=np.float64)
    v = valid_mask(p)

    def along(axis: int) -> np.ndarray:
        d = np.full(p.shape, np.nan)
        fwd = np.full(p.shape, np.nan)
        bwd = np.full(p.shape, np.nan)
        sl_hi = [slice(None)] * 2
        sl_lo = [slice(None)] * 2
        sl_hi[axis] = slice(1, None)
        sl_lo[axis] = slice(None, -1)
        diff = p[tuple(sl_hi)] - p[tuple(sl_lo)]
        ok = v[tuple(sl_hi)] & v[tuple(sl_lo)]
        diff[~ok] = np.nan
        fwd[tuple(sl_lo)] = diff  # p[i+1]-p[i] stored at i
        bwd[tuple(sl_hi)] = diff  # p[i]-p[i-1] stored at i
        both = np.isfinite(fwd).all(-1) & np.isfinite(bwd).all(-1)
        only_f = np.isfinite(fwd).all(-1) & ~both
        only_b = np.isfinite(bwd).all(-1) & ~both
        d[both] = 0.5 * (fwd[both] + bwd[both])
        d[only_f] = fwd[only_f]
        d[only_b] = bwd[only_b]
        d[~v] = np.nan
        return d

    return along(1), along(0)


def vertex_normals(points: np.ndarray) -> np.ndarray:
    """Unit normals n = normalize(dp/du x dp/dv) (u = column, v = row).

    Same orientation as the official C++ ``grid_normal`` (xv x yv, column
    difference cross row difference). NOTE the official *Python* package
    uses the opposite order (row x column), i.e. the opposite sign.
    NaN where not computable.
    """
    du, dv = grid_derivatives(points)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(norm > 1e-12, n / norm, np.nan)
    return n


def area_vox2(points: np.ndarray) -> float:
    """Surface area in voxel^2: each valid quad split into two triangles."""
    p = np.asarray(points, dtype=np.float64)
    qv = quad_valid_mask(p)
    p00, p01 = p[:-1, :-1][qv], p[:-1, 1:][qv]
    p10, p11 = p[1:, :-1][qv], p[1:, 1:][qv]
    a1 = 0.5 * np.linalg.norm(np.cross(p01 - p00, p10 - p00), axis=-1)
    a2 = 0.5 * np.linalg.norm(np.cross(p01 - p11, p10 - p11), axis=-1)
    return float(a1.sum() + a2.sum())


def area_cm2(points: np.ndarray, voxel_size_um: float) -> float:
    """Surface area in cm^2. ``voxel_size_um`` must be supplied by the caller
    (no default: the voxel size is scan-specific)."""
    if voxel_size_um is None or not (voxel_size_um > 0):
        raise ValueError("voxel_size_um must be a positive number supplied by the caller")
    um2_per_vox2 = float(voxel_size_um) ** 2
    return area_vox2(points) * um2_per_vox2 * 1e-8  # 1 cm^2 = 1e8 um^2


def edge_lengths(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lengths of u-edges (H, W-1) and v-edges (H-1, W); NaN if an end is invalid."""
    p = np.asarray(points, dtype=np.float64)
    v = valid_mask(p)
    lu = np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)
    lv = np.linalg.norm(p[1:] - p[:-1], axis=-1)
    lu[~(v[:, 1:] & v[:, :-1])] = np.nan
    lv[~(v[1:] & v[:-1])] = np.nan
    return lu, lv


def edge_fraction_within(points: np.ndarray, nominal_step: float, pct: float) -> float:
    """Fraction of valid grid edges whose length is within +-pct % of
    ``nominal_step`` (voxels). NaN if there are no valid edges."""
    lu, lv = edge_lengths(points)
    L = np.concatenate([lu[np.isfinite(lu)], lv[np.isfinite(lv)]])
    if L.size == 0:
        return math.nan
    return float(np.mean(np.abs(L - nominal_step) <= abs(pct) / 100.0 * nominal_step))


def sample_grid(points: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of an (H, W, 3) grid at fractional (row, col).

    A sample is valid iff it lies inside the grid and all four corners of its
    containing quad are valid (invalidity propagates); invalid samples get
    the sentinel. Samples exactly on the last row/column use the last quad.
    """
    p = np.asarray(points, dtype=np.float64)
    H, W = p.shape[:2]
    r = np.asarray(rows, dtype=np.float64)
    c = np.asarray(cols, dtype=np.float64)
    shape = np.broadcast(r, c).shape
    r = np.broadcast_to(r, shape).ravel()
    c = np.broadcast_to(c, shape).ravel()
    out = np.full((r.size, 3), SENTINEL, dtype=np.float64)
    eps = 1e-9
    inside = np.isfinite(r) & np.isfinite(c) & (r >= -eps) & (r <= H - 1 + eps) & (c >= -eps) & (c <= W - 1 + eps)
    idx = np.nonzero(inside)[0]
    if idx.size:
        rr = np.clip(r[idx], 0, H - 1)
        cc = np.clip(c[idx], 0, W - 1)
        j = np.minimum(np.floor(rr).astype(np.int64), H - 2)
        i = np.minimum(np.floor(cc).astype(np.int64), W - 2)
        t = (rr - j)[:, None]
        s = (cc - i)[:, None]
        qv = quad_valid_mask(p)[j, i]
        val = ((1 - s) * (1 - t) * p[j, i] + s * (1 - t) * p[j, i + 1]
               + (1 - s) * t * p[j + 1, i] + s * t * p[j + 1, i + 1])
        out[idx[qv]] = val[qv]
    return out.reshape(*shape, 3)
