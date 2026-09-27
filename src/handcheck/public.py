"""Read-only access to the public Vesuvius Challenge bucket over HTTPS
(standard library only): the root catalogue, tifxyz meshes, and nearest-voxel
or per-plane reads of uncompressed OME-Zarr (v2) levels.

Everything here reads the anonymous public bucket
https://vesuvius-challenge-open-data.s3.amazonaws.com (data licence: see the
catalogue entry of each volume, CC BY-NC 4.0 at the time of writing).
Only per-slice mask statistics and per-vertex samples are kept in memory;
nothing is rendered.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .level_mesh import CentreCurve

BUCKET_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def _get(key: str, byte_range: tuple[int, int] | None = None, retries: int = 5) -> bytes | None:
    """GET one object (or a byte range of it); None if it does not exist (missing zarr chunk = fill value)."""
    req = urllib.request.Request(f"{BUCKET_URL}/{urllib.parse.quote(key)}")
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                return gzip.decompress(b) if r.headers.get("Content-Encoding") == "gzip" else b
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed to fetch {key}: {err}")


def list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET_URL}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            x = r.read().decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", x)
        if not m:
            return keys
        token = m.group(1)


def catalogue(cache: Path | None = None) -> dict:
    """The root catalogue metadata.json (dict with 'samples')."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text())
    d = json.loads(_get("metadata.json"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(d))
    return d


def volume_entry(cat: dict, scroll: str, volume_id: str) -> dict:
    return cat["samples"][scroll]["volumes"][volume_id]


def volume_prefix(cat: dict, scroll: str, volume_id: str) -> str:
    return f"{scroll}/volumes/{volume_entry(cat, scroll, volume_id)['long_id']}"


def affine(cat: dict, scroll: str, frm: str, to: str) -> np.ndarray | None:
    """3x4 organizer affine (x, y, z, 1) -> (x', y', z') from volume ``frm`` to ``to``
    (inverted from the reverse entry if only that one is listed)."""
    for a, b, inv in ((frm, to, False), (to, frm, True)):
        for vt in cat["samples"][scroll]["sample"]["properties"].get("volume_transforms") or []:
            if vt["from_volume_id"] != a:
                continue
            for t in vt["transforms"]:
                if t["to_volume_id"] == b:
                    M = np.vstack([np.array(t["matrix"], float), [0, 0, 0, 1]])
                    return (np.linalg.inv(M) if inv else M)[:3]
    return None


def umbilicus(cat: dict, scroll: str, volume_id: str) -> tuple[CentreCurve | None, dict]:
    """Official umbilicus of ``scroll`` expressed on ``volume_id``: native, or mapped by the organizer affine."""
    for key in list_keys(f"{scroll}/representations/umbilicus/"):
        if not key.endswith(".json"):
            continue
        annotated_on = Path(key).name.split("-umbilicus-")[0]
        cp = json.loads(_get(key))["control_points"]
        P = np.array([[c["x"], c["y"], c["z"]] for c in cp], float)
        info = dict(file=key, annotated_on=annotated_on, n_points=len(cp))
        if annotated_on != volume_id:
            A = affine(cat, scroll, annotated_on, volume_id)
            if A is None:
                continue
            P = P @ A[:, :3].T + A[:, 3]
            info.update(mapped_to=volume_id, affine_det=float(np.linalg.det(A[:, :3])))
        return CentreCurve(z=P[:, 2], x=P[:, 0], y=P[:, 1]), info
    return None, {}


def fetch_meshes(scroll: str, volume_id: str, dest: Path, cat: dict | None = None, workers: int = 16) -> list[Path]:
    """Download the organizer tifxyz meshes of ``scroll`` expressed on ``volume_id``: the catalogue
    segments' tifxyz entries whose path is .../mesh/<id>-on-<volume_id>-<um>.tifxyz/. Cached."""
    cat = cat if cat is not None else catalogue()
    by_mesh: dict[tuple[str, str], list[str]] = {}
    for seg_id, seg in sorted(cat["samples"][scroll].get("segments", {}).items()):
        for d in seg.get("data", []):
            for o in d.get("origins", []):
                path = o["path"].rstrip("/")
                m = re.match(rf"^{scroll}/segments/([^/]+)/mesh/([^/]+-on-{volume_id}-[^/]+\.tifxyz)$", path)
                if m:
                    keys = [k for k in list_keys(path + "/") if Path(k).name in ("x.tif", "y.tif", "z.tif", "meta.json", "mask.tif")]
                    by_mesh[(m.group(1), m.group(2))] = keys
    out = []
    jobs = []
    for (seg, name), keys in sorted(by_mesh.items()):
        d = dest / seg / name
        d.mkdir(parents=True, exist_ok=True)
        jobs += [(k, d / Path(k).name) for k in keys if not (d / Path(k).name).exists()]
        out.append(d)

    def dl(job):
        k, f = job
        tmp = f.with_suffix(f.suffix + ".part")
        tmp.write_bytes(_get(k))
        tmp.replace(f)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(dl, jobs))
    return out


class ZarrLevel:
    """Uncompressed, filter-free, C-order OME-Zarr v2 level on the public bucket."""

    def __init__(self, prefix: str, level: int):
        self.prefix, self.level, self.f = prefix.rstrip("/"), level, 2 ** level
        za = json.loads(_get(f"{self.prefix}/{level}/.zarray"))
        if za.get("compressor") is not None or za.get("filters") or za.get("order", "C") != "C":
            raise NotImplementedError("only uncompressed, filter-free, C-order zarr v2 levels are supported")
        self.shape = tuple(za["shape"])
        self.chunks = tuple(za["chunks"])
        self.dtype = np.dtype(za["dtype"])
        self.sep = za.get("dimension_separator", ".")
        self.fill = za.get("fill_value") or 0
        self._planes: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _key(self, iz: int, iy: int, ix: int) -> str:
        return f"{self.prefix}/{self.level}/" + self.sep.join(map(str, (iz, iy, ix)))

    def plane(self, z: int, iy: int, ix: int) -> np.ndarray:
        """One z plane (cy, cx) of chunk (z // cz, iy, ix), by a byte-range read. Cached."""
        cz, cy, cx = self.chunks
        iz, zz = divmod(z, cz)
        k = (iz, zz, iy, ix)
        if k not in self._planes:
            if len(self._planes) > 20000:
                self._planes.clear()
            pb = cy * cx * self.dtype.itemsize
            b = _get(self._key(iz, iy, ix), (zz * pb, (zz + 1) * pb - 1))
            self._planes[k] = (np.full((cy, cx), self.fill, self.dtype) if b is None
                               else np.frombuffer(b, self.dtype).reshape(cy, cx))
        return self._planes[k]

    def prefetch(self, keys: list[tuple[int, int, int]], workers: int = 16) -> None:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda k: self.plane(*k), keys))

    def full_plane(self, z: int, workers: int = 16) -> np.ndarray:
        Z, Y, X = self.shape
        cz, cy, cx = self.chunks
        ny, nx = -(-Y // cy), -(-X // cx)
        self.prefetch([(z, iy, ix) for iy in range(ny) for ix in range(nx)], workers)
        out = np.zeros((ny * cy, nx * cx), self.dtype)
        for iy in range(ny):
            for ix in range(nx):
                out[iy * cy:(iy + 1) * cy, ix * cx:(ix + 1) * cx] = self._planes.pop((z // cz, z % cz, iy, ix))
        return out[:Y, :X]

    @property
    def shape_level0(self) -> tuple[int, int, int]:
        return tuple(s * self.f for s in self.shape)  # type: ignore[return-value]

    def sample(self, P: np.ndarray) -> np.ndarray:
        """Nearest level voxel values at (N, 3) level-0 (x, y, z) points inside the volume."""
        Z, Y, X = self.shape
        cz, cy, cx = self.chunks
        idx = np.floor(np.asarray(P, float) / self.f).astype(np.int64)
        xi, yi, zi = (np.clip(idx[:, 0], 0, X - 1), np.clip(idx[:, 1], 0, Y - 1), np.clip(idx[:, 2], 0, Z - 1))
        need = sorted({(int(z), int(y) // cy, int(x) // cx) for z, y, x in zip(zi, yi, xi)})
        self.prefetch(need)
        return np.array([self.plane(int(z), int(y) // cy, int(x) // cx)[int(y) % cy, int(x) % cx]
                         for z, y, x in zip(zi, yi, xi)])


def slice_stats(mask: np.ndarray) -> dict:
    """Centroid, area and minor/major axis ratio of one binary slice (y, x)."""
    yy, xx = np.nonzero(mask)
    n = int(yy.size)
    if n == 0:
        return dict(n=0, cx=math.nan, cy=math.nan, axis_ratio=math.nan)
    cx, cy = float(xx.mean()), float(yy.mean())
    C = np.cov(np.stack([xx - cx, yy - cy])) if n > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(C))
    return dict(n=n, cx=cx, cy=cy, axis_ratio=float(math.sqrt(max(ev[0], 1e-12) / max(ev[1], 1e-12))))


def centroid_track(z_level: ZarrLevel, zstride: int = 8) -> list[dict]:
    """Per-slice CT-mask (value > 0) centroid in level-0 voxels, every ``zstride``-th level slice."""
    f = z_level.f
    rows = []
    for z in range(0, z_level.shape[0], zstride):
        st = slice_stats(z_level.full_plane(z) > 0)
        rows.append(dict(z=z * f + (f - 1) / 2, n=st["n"], axis_ratio=st["axis_ratio"],
                         cx=(st["cx"] * f + (f - 1) / 2) if st["n"] else None,
                         cy=(st["cy"] * f + (f - 1) / 2) if st["n"] else None))
    return rows


def centroid_curve(rows: list[dict], sigma: float = 256.0) -> tuple[CentreCurve | None, dict]:
    """z-smoothed centroid curve; slices whose mask area is < 0.2x or > 2.5x the median are dropped
    (unmasked full field of view, near-empty ends)."""
    sl = [r for r in rows if r["cx"] is not None and r["n"] > 0]
    if len(sl) < 3:
        return None, dict(error="too few slices")
    area = np.array([r["n"] for r in sl], float)
    med = np.median(area)
    keep = (area > 0.2 * med) & (area < 2.5 * med)
    z = np.array([r["z"] for r in sl], float)[keep]
    cx = np.array([r["cx"] for r in sl], float)[keep]
    cy = np.array([r["cy"] for r in sl], float)[keep]
    w = np.exp(-0.5 * ((z[:, None] - z[None, :]) / sigma) ** 2)
    w /= w.sum(1, keepdims=True)
    ar = np.array([r["axis_ratio"] for r in sl], float)[keep]
    return CentreCurve(z=z, x=w @ cx, y=w @ cy), dict(n_slices=len(sl), n_used=int(keep.sum()), sigma_vox=sigma,
                                                     axis_ratio_median=float(np.median(ar)))
