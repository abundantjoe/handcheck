"""Levelling (iso-z horizontal re-parameterisation), handedness and outward
normals for tifxyz grids.

Grid convention: ``points[v, u] = (x, y, z)`` in scanner voxel coordinates,
u = column (image x, to the right), v = row (image y, downward); invalid
vertices are (-1, -1, -1). See ``tifxyz.py``.

Levelling method
----------------
1. z-gradient direction. For every valid quad the cell-centred z gradient in
   grid units is ``g = (dz/du, dz/dv)`` (average of the two opposite edge
   differences). The angle ``theta = atan2(dz/dv, dz/du)`` of the direction in
   which z increases is estimated robustly (default ``method="median"``): the
   circular mean of the unit gradient directions is taken as a centre, and
   ``theta`` = centre + median of the wrapped angular deviations (two passes).
   ``method="lsq"`` instead fits the plane ``z = a u + b v + c`` over all
   valid vertices by least squares and uses ``atan2(b, a)``. The median-
   absolute-deviation of the per-cell angles is returned as a diagnostic: a
   single global rotation only levels a tile whose iso-z direction is
   uniform, so large-spread meshes should be levelled tile by tile.
2. Proper rotation (det = +1, so handedness is untouched). New axes in the
   (u, v) plane: ``e_v = (cos theta, sin theta)`` (z increases along +v'),
   ``e_u = (sin theta, -cos theta)``. Hence rows of the output have constant
   z and z increases with the row index.
3. Arc-length spacing. The metric along each new axis
   ``s = |dp/du * e[0] + dp/dv * e[1]|`` (voxels per grid unit) is taken as
   the median over valid quads; the new grid spacing in grid units is
   ``step_vox / s``. This yields step_vox arc-length spacing when the input
   parameterisation is (near-)isometric, as tracer output is; the realised
   edge lengths should always be checked with
   ``tifxyz.edge_fraction_within``.
4. Resampling. Every output node is mapped back to fractional source (row,
   col) and xyz is bilinearly interpolated over the source grid; a node is
   valid only if its source quad has four valid corners (invalid propagates).

Handedness
----------
``chirality = sign(det[dp/du, dp/dv, n_out])`` in scanner (x, y, z).
When an image is displayed with u to the right and v downward, ``dp/du x
dp/dv`` points INTO the screen, away from the viewer. So chirality -1 means
the viewer is on the outward (away-from-umbilicus) side of the sheet, +1 on
the umbilicus side. The sign that gives legible text is a PARAMETER
(``convention_sign``) to be validated on known-text scrolls. Note: a
reflected reconstruction (e.g. reversed slice order = z flip) swaps which face
appears legible and so needs the opposite sign; a rigid rotation of the scroll
(mounted upside down) does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

from .tifxyz import (
    TifxyzMesh,
    grid_derivatives,
    quad_valid_mask,
    sample_grid,
    valid_mask,
)


# ---------------------------------------------------------------------------
# centre curve (umbilicus) and outward directions
# ---------------------------------------------------------------------------

@dataclass
class CentreCurve:
    """Umbilicus c(z) given as samples (z_i, x_i, y_i); linear interpolation,
    clamped (constant) beyond the ends."""

    z: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        z = np.atleast_1d(np.asarray(self.z, dtype=np.float64))
        order = np.argsort(z)
        self.z = z[order]
        self.x = np.atleast_1d(np.asarray(self.x, dtype=np.float64))[order]
        self.y = np.atleast_1d(np.asarray(self.y, dtype=np.float64))[order]

    @classmethod
    def straight(cls, cx: float, cy: float) -> "CentreCurve":
        return cls(z=np.array([0.0]), x=np.array([cx]), y=np.array([cy]))

    def __call__(self, zq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zq = np.asarray(zq, dtype=np.float64)
        if self.z.size == 1:
            return np.full_like(zq, self.x[0]), np.full_like(zq, self.y[0])
        return np.interp(zq, self.z, self.x), np.interp(zq, self.z, self.y)


CentreLike = Union[CentreCurve, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]]


def outward_directions(points: np.ndarray, centre: CentreLike) -> np.ndarray:
    """Unit r_out = (p - c(z)) projected perpendicular to z; NaN where invalid
    or where the vertex is (numerically) on the centre curve."""
    p = np.asarray(points, dtype=np.float64)
    v = valid_mask(p)
    cx, cy = centre(p[..., 2])
    r = np.stack([p[..., 0] - cx, p[..., 1] - cy, np.zeros(p.shape[:2])], axis=-1)
    n = np.linalg.norm(r, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(n > 1e-9, r / n, np.nan)
    r[~v] = np.nan
    return r


@dataclass
class NormalSignReport:
    sign: int  # +1 normals point outward, -1 inward, 0 undetermined
    frac_clear: float  # fraction of valid vertices with |cos| > cos_thresh
    frac_positive: float  # among clear vertices, fraction with cos > 0
    ambiguous: bool
    n_valid: int
    cos: np.ndarray = field(repr=False)  # (H, W) n . r_out, NaN if undefined


def normal_sign_report(
    points: np.ndarray,
    normals: np.ndarray,
    centre: CentreLike,
    *,
    cos_thresh: float = 0.3,
    min_clear_fraction: float = 0.8,
    min_agreement: float = 0.95,
) -> NormalSignReport:
    """Classify the normal orientation relative to the outward direction.

    A vertex is *clear* if |n . r_out| > cos_thresh. The mesh is flagged
    ambiguous when fewer than ``min_clear_fraction`` of its valid vertices are
    clear, or the clear vertices disagree (majority share < min_agreement).
    """
    r = outward_directions(points, centre)
    n = np.asarray(normals, dtype=np.float64)
    cos = np.einsum("...k,...k->...", n, r)
    fin = np.isfinite(cos)
    n_valid = int(fin.sum())
    clear = fin & (np.abs(np.where(fin, cos, 0.0)) > cos_thresh)
    n_clear = int(clear.sum())
    frac_clear = n_clear / n_valid if n_valid else 0.0
    frac_pos = float((cos[clear] > 0).mean()) if n_clear else math.nan
    if n_clear == 0:
        sign = 0
    else:
        sign = 1 if frac_pos > 0.5 else (-1 if frac_pos < 0.5 else 0)
    agreement = max(frac_pos, 1 - frac_pos) if n_clear else 0.0
    ambiguous = (sign == 0) or (frac_clear < min_clear_fraction) or (agreement < min_agreement)
    return NormalSignReport(sign, frac_clear, frac_pos, bool(ambiguous), n_valid, cos)


def orient_normals_outward(
    points: np.ndarray, normals: np.ndarray, centre: CentreLike, **kw
) -> tuple[np.ndarray, NormalSignReport]:
    """Flip the whole normal field (never per vertex) so it points outward.
    Returns (normals, report of the INPUT normals)."""
    rep = normal_sign_report(points, normals, centre, **kw)
    n = np.asarray(normals, dtype=np.float64)
    return (-n if rep.sign < 0 else n.copy()), rep


# ---------------------------------------------------------------------------
# levelling
# ---------------------------------------------------------------------------

def _cell_z_gradients(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)
    qv = quad_valid_mask(p)
    z = p[..., 2]
    gu = 0.5 * ((z[:-1, 1:] - z[:-1, :-1]) + (z[1:, 1:] - z[1:, :-1]))
    gv = 0.5 * ((z[1:, :-1] - z[:-1, :-1]) + (z[1:, 1:] - z[:-1, 1:]))
    return gu[qv], gv[qv]


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def estimate_level_angle(points: np.ndarray, method: str = "median") -> tuple[float, float]:
    """Angle (degrees, from +u toward +v) of the direction in which z
    increases in the (u, v) grid, and the MAD spread (degrees) of the
    per-cell directions. See module docstring."""
    p = np.asarray(points, dtype=np.float64)
    gu, gv = _cell_z_gradients(p)
    mag = np.hypot(gu, gv)
    keep = mag > 1e-9
    if keep.sum() == 0:
        raise ValueError("no valid quad with a non-zero z gradient")
    th = np.arctan2(gv[keep], gu[keep])
    if method == "median":
        centre = math.atan2(np.sin(th).sum(), np.cos(th).sum())
        for _ in range(2):
            centre = centre + float(np.median(_wrap(th - centre)))
        theta = centre
    elif method == "lsq":
        v = valid_mask(p)
        vv, uu = np.nonzero(v)
        A = np.stack([uu, vv, np.ones_like(uu)], axis=1).astype(np.float64)
        (a, b, _), *_ = np.linalg.lstsq(A, p[vv, uu, 2], rcond=None)
        if abs(a) + abs(b) < 1e-12:
            raise ValueError("least-squares z plane is flat")
        theta = math.atan2(b, a)
    else:
        raise ValueError(f"unknown method {method!r}")
    spread = float(np.degrees(np.median(np.abs(_wrap(th - theta)))))
    return math.degrees(_wrap(np.array(theta)).item()), spread


@dataclass
class LevelResult:
    points: np.ndarray  # (H', W', 3) float32, sentinel for invalid
    src_u: np.ndarray  # (H', W') fractional source column of each node
    src_v: np.ndarray  # (H', W') fractional source row of each node
    angle_deg: float  # z-gradient direction in the source (u, v) grid
    spread_deg: float  # MAD of per-cell gradient directions
    e_u: np.ndarray  # new +u' axis in source (u, v) grid units
    e_v: np.ndarray  # new +v' axis (z increases along it)
    origin: tuple[float, float]  # (U'0, V'0): rotated coords of node (0, 0)
    du: float  # source grid units per output column
    dv: float  # source grid units per output row
    step_vox: float
    metric: tuple[float, float]  # vox per source grid unit along e_u, e_v

    @property
    def valid(self) -> np.ndarray:
        return valid_mask(self.points)

    def to_level_index(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Source grid (u=col, v=row) -> fractional output (row, col)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        U = u * self.e_u[0] + v * self.e_u[1]
        V = u * self.e_v[0] + v * self.e_v[1]
        return (V - self.origin[1]) / self.dv, (U - self.origin[0]) / self.du

    def map_back(self, source_shape: tuple[int, int]) -> np.ndarray:
        """Sample the levelled surface at every source grid vertex (bilinear
        over the levelled grid); sentinel where the levelled grid is invalid."""
        H, W = source_shape
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        r, c = self.to_level_index(uu, vv)
        return sample_grid(self.points, r, c).astype(np.float32)

    def to_mesh(self, uuid: str, meta: dict | None = None) -> TifxyzMesh:
        s = 1.0 / self.step_vox
        m = dict(meta or {})
        m.setdefault("handcheck_level_angle_deg", self.angle_deg)
        return TifxyzMesh(points=self.points.copy(), scale=(s, s), uuid=uuid, meta=m)


def _axis_metric(points: np.ndarray, e: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    qv = quad_valid_mask(p)
    dpdu = 0.5 * ((p[:-1, 1:] - p[:-1, :-1]) + (p[1:, 1:] - p[1:, :-1]))[qv]
    dpdv = 0.5 * ((p[1:, :-1] - p[:-1, :-1]) + (p[1:, 1:] - p[:-1, 1:]))[qv]
    s = np.linalg.norm(dpdu * e[0] + dpdv * e[1], axis=-1)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        raise ValueError("cannot estimate the grid metric (no valid quads)")
    return float(np.median(s))


def level_mesh(
    points_grid: np.ndarray,
    step_vox: float = 20.0,
    *,
    angle_deg: float | None = None,
    method: str = "median",
    crop: bool = True,
) -> LevelResult:
    """Rotate the (u, v) parameterisation so iso-z curves are rows, and
    resample on a regular grid with ``step_vox`` arc-length spacing.

    ``angle_deg`` overrides the estimated z-gradient direction (e.g. to share
    one rotation between tiles). See the module docstring for the method.
    """
    p = np.asarray(points_grid, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError("points_grid must be (H, W, 3)")
    if step_vox <= 0:
        raise ValueError("step_vox must be > 0")
    v_ok = valid_mask(p)
    if not v_ok.any():
        raise ValueError("no valid vertices")
    if angle_deg is None:
        angle_deg, spread = estimate_level_angle(p, method=method)
    else:
        try:
            _, spread = estimate_level_angle(p, method="median")
        except ValueError:
            spread = math.nan
    th = math.radians(angle_deg)
    e_v = np.array([math.cos(th), math.sin(th)])
    e_u = np.array([math.sin(th), -math.cos(th)])  # det[e_u, e_v] = +1

    s_u = _axis_metric(p, e_u)
    s_v = _axis_metric(p, e_v)
    du = step_vox / s_u
    dv = step_vox / s_v

    vv, uu = np.nonzero(v_ok)
    U = uu * e_u[0] + vv * e_u[1]
    V = uu * e_v[0] + vv * e_v[1]
    U0, V0 = float(U.min()), float(V.min())
    nU = int(math.floor((U.max() - U0) / du + 1e-9)) + 1
    nV = int(math.floor((V.max() - V0) / dv + 1e-9)) + 1
    Ug = U0 + du * np.arange(nU)
    Vg = V0 + dv * np.arange(nV)
    VV, UU = np.meshgrid(Vg, Ug, indexing="ij")
    src_u = UU * e_u[0] + VV * e_v[0]
    src_v = UU * e_u[1] + VV * e_v[1]
    out = sample_grid(p, src_v, src_u)

    if crop:
        ok = valid_mask(out)
        if ok.any():
            rows = np.nonzero(ok.any(axis=1))[0]
            cols = np.nonzero(ok.any(axis=0))[0]
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            out = out[r0:r1, c0:c1]
            src_u = src_u[r0:r1, c0:c1]
            src_v = src_v[r0:r1, c0:c1]
            U0 += c0 * du
            V0 += r0 * dv
    return LevelResult(
        points=out.astype(np.float32),
        src_u=src_u,
        src_v=src_v,
        angle_deg=float(angle_deg),
        spread_deg=float(spread),
        e_u=e_u,
        e_v=e_v,
        origin=(U0, V0),
        du=float(du),
        dv=float(dv),
        step_vox=float(step_vox),
        metric=(s_u, s_v),
    )


def iso_z_tilt_deg(points: np.ndarray) -> float:
    """Angle (deg) between the mean z-gradient and the +v (row) axis: 0 when
    iso-z curves are exactly horizontal rows with z increasing downward."""
    gu, gv = _cell_z_gradients(points)
    if gu.size == 0:
        return math.nan
    return abs(math.degrees(math.atan2(float(np.mean(gu)), float(np.mean(gv)))))


# ---------------------------------------------------------------------------
# handedness / reading orientation
# ---------------------------------------------------------------------------

OutwardLike = Union[np.ndarray, CentreCurve, Callable]


def _resolve_outward(points: np.ndarray, outward: OutwardLike) -> np.ndarray:
    p = np.asarray(points)
    if isinstance(outward, np.ndarray) or isinstance(outward, (list, tuple)):
        o = np.asarray(outward, dtype=np.float64)
        if o.shape == (3,):
            return np.broadcast_to(o, p.shape).copy()
        if o.shape != p.shape:
            raise ValueError(f"outward normals shape {o.shape} != points shape {p.shape}")
        return o
    return outward_directions(p, outward)


def chirality_map(points: np.ndarray, outward_normals: OutwardLike) -> np.ndarray:
    """Per-vertex normalised det[dp/du, dp/dv, n_out] (a signed sine); NaN
    where undefined."""
    du, dv = grid_derivatives(points)
    n = _resolve_outward(points, outward_normals)
    det = np.einsum("...k,...k->...", np.cross(du, dv), n)
    den = (np.linalg.norm(du, axis=-1) * np.linalg.norm(dv, axis=-1) * np.linalg.norm(n, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, det / den, np.nan)


def chirality(points_grid: np.ndarray, outward_normals: OutwardLike, *, min_abs: float = 0.1) -> int:
    """sign(det[dp/du, dp/dv, n_out]) by majority vote over vertices with
    |normalised det| > min_abs; 0 if no vertex qualifies or a tie."""
    c = chirality_map(points_grid, outward_normals)
    c = c[np.isfinite(c) & (np.abs(np.nan_to_num(c)) > min_abs)]
    if c.size == 0:
        return 0
    s = int(np.sign(np.sum(np.sign(c))))
    return s


@dataclass
class OrientResult:
    points: np.ndarray
    flip_u: bool
    flip_v: bool
    chirality_before: int  # of the input grid
    chirality_after: int  # of the returned grid (== convention_sign)
    z_trend: float  # median dz per row step after orientation
    levelled_ok: bool  # z varies mostly along rows (v) rather than columns (u)
    mesh: TifxyzMesh | None = None

    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Apply the same flips to any (H, W, ...) array (channels, normals,
        rendered images with the grid in their first two axes)."""
        a = np.asarray(arr)
        if self.flip_v:
            a = a[::-1]
        if self.flip_u:
            a = a[:, ::-1]
        return np.ascontiguousarray(a)


def orient_for_reading(
    mesh: np.ndarray | TifxyzMesh,
    z_ascending: bool,
    convention_sign: int,
    *,
    outward: OutwardLike,
) -> OrientResult:
    """Flip v and/or u (never rotate) so that

    * rows run with z increasing (``z_ascending=True``: image top = low z) or
      decreasing (``z_ascending=False``), and
    * ``chirality(points, outward) == convention_sign`` (mirror u otherwise).

    ``convention_sign`` (+1/-1) is an unvalidated parameter; see the module
    docstring for its geometric meaning. ``outward`` is a CentreCurve (or
    callable c(z)), a per-vertex (H, W, 3) outward field or a single vector.
    Run ``level_mesh`` first so that z varies along rows.
    """
    if convention_sign not in (1, -1):
        raise ValueError("convention_sign must be +1 or -1")
    pts = mesh.points if isinstance(mesh, TifxyzMesh) else np.asarray(mesh)
    pts = np.asarray(pts, dtype=np.float32)
    out_field = _resolve_outward(pts, outward)
    before = chirality(pts, out_field)
    if before == 0:
        raise ValueError("chirality undetermined (degenerate mesh or outward field)")

    v_ok = valid_mask(pts)
    z = pts[..., 2].astype(np.float64)
    ok_v = v_ok[1:] & v_ok[:-1]
    ok_u = v_ok[:, 1:] & v_ok[:, :-1]
    dz_v = (z[1:] - z[:-1])[ok_v]
    dz_u = (z[:, 1:] - z[:, :-1])[ok_u]
    trend_v = float(np.median(dz_v)) if dz_v.size else 0.0
    trend_u = float(np.median(np.abs(dz_u))) if dz_u.size else 0.0
    if trend_v == 0.0:
        raise ValueError("z does not vary along rows; level the mesh first")
    levelled_ok = abs(trend_v) > trend_u

    flip_v = (trend_v > 0) != bool(z_ascending)
    if flip_v:  # a single-axis flip reverses the chirality
        pts = pts[::-1]
        out_field = out_field[::-1]
    current = -before if flip_v else before
    flip_u = current != convention_sign
    if flip_u:
        pts = pts[:, ::-1]
        out_field = out_field[:, ::-1]
    pts = np.ascontiguousarray(pts)
    after = chirality(pts, out_field)
    ok_after = valid_mask(pts)
    trend_after = float(np.median(np.diff(pts[..., 2].astype(np.float64), axis=0)[ok_after[1:] & ok_after[:-1]]))
    res = OrientResult(pts, bool(flip_u), bool(flip_v), before, after, trend_after, bool(levelled_ok))
    if isinstance(mesh, TifxyzMesh):
        res.mesh = TifxyzMesh(points=pts.copy(), scale=mesh.scale, uuid=mesh.uuid, meta=dict(mesh.meta))
    return res
