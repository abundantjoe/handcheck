"""Geometry-only reading orientation (handedness) of a tifxyz mesh.

Rule (validated on synthetic scrolls; see tests/test_rule.py and the README):

* F1  The recto (written face) is rolled facing the umbilicus, so text is
  read from the umbilicus side.
* F2  With u to the right and v downward, dP/du x dP/dv points into the
  screen, so the viewer is on the umbilicus side iff
  det[dP/du, dP/dv, r_out] > 0 in a right-handed frame.
* F3  The index frame (x, y, z) has parity P = -1 if the root catalogue says
  ``left_handed_coordinates`` is true, +1 if false (0 = unknown -> FLAG).
  Legible iff chirality == P. The acquisition z direction plays no part.
* F4  Fragments use n_out = -text_side instead of r_out.
* Up/down: ``z_direction_is_top_to_bottom`` True -> image top = low z,
  False -> high z, None -> unknown (only a 180 degree ambiguity).

r_out sources (independent): official umbilicus, else CT-mask centroid
(level 4, z-smoothed; a proxy for the umbilicus, so informational only when
an umbilicus exists), plus the mesh normal-line centre (valid only if its
2x2 systems are well conditioned). Gates, in order:

1. parity unknown                         -> PARITY_UNKNOWN (FLAG)
2. frame check fails (on zero CT > 5 % or outside the volume > 1 %)
                                          -> FRAME_UNVERIFIED (FLAG; do not read)
3. no valid source                        -> FLAG
4. valid sources disagree on the whole-mesh sign -> FLAG (both mirror states)
5. vertices reversed under every valid source (fold / sheet jump):
   <= 20 % of the clear vertices -> masked out of ``keep`` (read the rest);
   >  20 %                          -> FLAG (both mirror states)
Confidence: HIGH needs >= 2 valid sources, each agreeing on >= 0.98 of its
clear non-reversed vertices, pairwise per-vertex agreement >= 0.95 and
reversed <= 2 %. MEDIUM: every source >= 0.9 and pairwise >= 0.8, or one
source >= 0.95 with reversed <= 2 %. Otherwise FLAG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np
from scipy.ndimage import binary_dilation

from .level_mesh import CentreCurve, chirality_map
from .tifxyz import TifxyzMesh, valid_mask, vertex_normals

AXIS_VEC = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}

# thresholds (see module docstring and the README, section Validation)
FRAME_MAX_ZERO = 0.05
FRAME_MAX_OUTSIDE = 0.01
COND_MAX = 10.0
MIN_ABS_DET = 0.1
REVERSED_MASK_MAX = 0.20
REVERSED_CLEAN_MAX = 0.02


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def frame_parity(zattrs: dict | None, props: dict | None) -> tuple[int, list[str]]:
    """Parity of the named (x, y, z) index frame and the notes that explain it."""
    notes: list[str] = []
    axes: list[str] = []
    try:
        axes = [a["name"] for a in (zattrs or {})["multiscales"][0]["axes"]]
    except (KeyError, IndexError, TypeError):
        notes.append("zarr axes unavailable")
    if sorted(axes) != ["x", "y", "z"]:
        notes.append(f"axes {axes} not a permutation of x, y, z")
        return 0, notes
    lh = (props or {}).get("left_handed_coordinates")
    if lh is True:
        return -1, notes
    if lh is False:
        return 1, notes
    notes.append("left_handed_coordinates missing")
    return 0, notes


def text_side_vector(text_side: str | None) -> np.ndarray | None:
    if not isinstance(text_side, str):
        return None
    s = text_side.strip().lower()
    ax = s.lstrip("+-")
    if ax not in AXIS_VEC:
        return None
    return (-1.0 if s.startswith("-") else 1.0) * AXIS_VEC[ax]


@dataclass
class VolumeInfo:
    """What the rule needs to know about the volume a mesh lives in."""

    parity: int
    z_top_to_bottom: bool | None
    sample_type: str = "scroll"  # or "fragment"
    text_side: np.ndarray | None = None
    umbilicus: CentreCurve | None = None
    centroid: CentreCurve | None = None
    name: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, zattrs: dict | None, props: dict | None, *, sample_type: str = "scroll",
                      umbilicus: CentreCurve | None = None, centroid: CentreCurve | None = None,
                      name: str = "") -> "VolumeInfo":
        parity, notes = frame_parity(zattrs, props)
        tb = (props or {}).get("z_direction_is_top_to_bottom")
        return cls(parity=parity, z_top_to_bottom=(None if tb is None else bool(tb)), sample_type=sample_type,
                   text_side=text_side_vector((props or {}).get("text_side")), umbilicus=umbilicus,
                   centroid=centroid, name=name, notes=notes)


@dataclass
class FrameCheck:
    """Fractions of (sampled) mesh vertices on zero CT and outside the
    volume, measured in the mesh's declared volume frame."""

    frac_zero_ct: float
    frac_outside: float

    @property
    def ok(self) -> bool:
        return self.frac_zero_ct <= FRAME_MAX_ZERO and self.frac_outside <= FRAME_MAX_OUTSIDE


def frame_check_from_volume(points: np.ndarray, sampler: Callable[[np.ndarray], np.ndarray],
                            shape_zyx: tuple[int, int, int], stride: int = 4) -> FrameCheck:
    """FrameCheck from any sampler returning CT values at (N, 3) level-0
    (x, y, z) points (e.g. nearest voxel of a low pyramid level)."""
    p = np.asarray(points, np.float64)
    P = p[valid_mask(p)][::stride]
    Z, Y, X = shape_zyx
    inside = (P[:, 0] >= 0) & (P[:, 0] < X) & (P[:, 1] >= 0) & (P[:, 1] < Y) & (P[:, 2] >= 0) & (P[:, 2] < Z)
    v = np.zeros(len(P))
    if inside.any():
        v[inside] = sampler(P[inside])
    return FrameCheck(frac_zero_ct=float((v == 0).mean()), frac_outside=float(1 - inside.mean()))


# ---------------------------------------------------------------------------
# mesh-intrinsic centre
# ---------------------------------------------------------------------------

def normal_line_centre(points: np.ndarray, *, z_bin_vox: float = 400.0, min_vertices: int = 50,
                       stride: int = 2) -> tuple[CentreCurve | None, float]:
    """Least-squares intersection of the xy-projected vertex normal lines per
    z band (centre of curvature; no CT). Returns (curve, median condition
    number of the per-band 2x2 systems); (None, inf) if undefined. A large
    condition number means near-parallel normals (flat or radial sheet), so
    the centre is ill-defined."""
    p = np.asarray(points, np.float64)[::stride, ::stride]
    n = vertex_normals(np.asarray(points, np.float64))[::stride, ::stride]
    ok = valid_mask(p) & np.isfinite(n).all(-1)
    nxy = n[..., :2]
    ok &= np.linalg.norm(nxy, axis=-1) > 0.3
    P, N = p[ok], nxy[ok]
    if len(P) < min_vertices:
        return None, math.inf
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    z = P[:, 2]
    edges = np.arange(z.min(), z.max() + z_bin_vox, z_bin_vox)
    zc, xc, yc, conds = [], [], [], []
    bands = [(a, b) for a, b in zip(edges[:-1], edges[1:])] or [(z.min(), z.max() + 1)]
    for a, b in bands:
        sel = (z >= a) & (z < b)
        if sel.sum() < min_vertices:
            continue
        Pr = np.eye(2)[None] - N[sel][:, :, None] * N[sel][:, None, :]
        A = Pr.sum(0)
        rhs = np.einsum("nij,nj->i", Pr, P[sel, :2])
        cond = float(np.linalg.cond(A))
        if not np.isfinite(cond) or cond > 1e12:  # parallel normals: no centre in this band
            conds.append(math.inf)
            continue
        c = np.linalg.solve(A, rhs)
        zc.append(0.5 * (a + b))
        xc.append(c[0])
        yc.append(c[1])
        conds.append(cond)
    if not zc:
        return None, math.inf
    return CentreCurve(z=np.array(zc), x=np.array(xc), y=np.array(yc)), float(np.median(conds))


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

@dataclass
class HandedDecision:
    state: str  # OK | FLAG | FRAME_UNVERIFIED | PARITY_UNKNOWN
    confidence: str  # HIGH | MEDIUM | FLAG
    convention_sign: int
    chirality: int  # consensus whole-mesh sign of the INPUT grid (0 if none)
    mirror_u: bool | None  # flip columns to read unmirrored (given flip_v; see rot180_equivalent)
    flip_v: bool | None  # flip rows to put the text top at the image top; None = up/down unknown
    rot180_equivalent: bool  # True when up/down is unknown: (flip_v=True, mirror_u=not m) is the other upright candidate
    keep: np.ndarray | None  # vertices to read (reversed patches removed)
    reversed_frac: float
    sources: dict[str, Any]
    flags: list[str]

    @property
    def readable(self) -> bool:
        return self.state == "OK" and self.confidence in ("HIGH", "MEDIUM")

    def apply(self, arr: np.ndarray, *, flip_v: bool | None = None) -> np.ndarray:
        """Apply the decided flips to any (H, W, ...) array on the mesh grid.
        When up/down is unknown pass ``flip_v`` explicitly (the mirror state is
        adjusted so the result is unmirrored either way)."""
        if self.mirror_u is None:
            raise ValueError(f"no orientation decided (state {self.state}, confidence {self.confidence})")
        fv = self.flip_v if self.flip_v is not None else bool(flip_v)
        mu = self.mirror_u if (self.flip_v is not None or not fv) else (not self.mirror_u)
        a = np.asarray(arr)
        if fv:
            a = a[::-1]
        if mu:
            a = a[:, ::-1]
        return np.ascontiguousarray(a)


def _flagged(state, conv, flags, sources, rev=math.nan, chir=0, keep=None) -> HandedDecision:
    return HandedDecision(state=state, confidence="FLAG", convention_sign=conv, chirality=chir, mirror_u=None,
                          flip_v=None, rot180_equivalent=False, keep=keep, reversed_frac=rev, sources=sources, flags=flags)


def orient_for_reading(mesh: np.ndarray | TifxyzMesh, volume: VolumeInfo, *, frame: FrameCheck | None = None,
                       keep: np.ndarray | None = None) -> HandedDecision:
    """Decide (mirror_u, flip_v, confidence) for reading ``mesh`` (its grid as
    given, u = column, v = row). ``frame``: result of the frame check in the
    declared volume (None = not checked, recorded as a flag). ``keep``:
    optional vertex mask (e.g. a region to exclude); excluded vertices
    never vote."""
    p = np.asarray(mesh.points if isinstance(mesh, TifxyzMesh) else mesh, np.float64)
    ok = valid_mask(p)
    if keep is not None:
        ok &= keep
    conv = int(volume.parity)
    flags: list[str] = list(volume.notes)
    sources: dict[str, Any] = {}
    if conv == 0:
        return _flagged("PARITY_UNKNOWN", 0, flags + ["frame parity unknown"], sources)
    if frame is None:
        flags.append("frame not checked")
    elif not frame.ok:
        return _flagged("FRAME_UNVERIFIED", conv, flags + [
            f"mesh not on the scroll in its declared frame: zero CT {frame.frac_zero_ct:.3f}, "
            f"outside {frame.frac_outside:.3f}"], dict(frame=vars(frame)))

    # r_out sources
    fields: dict[str, Any] = {}
    if volume.sample_type == "fragment":
        if volume.text_side is None:
            return _flagged("FLAG", conv, flags + ["fragment without text_side"], sources)
        fields["text_side"] = -volume.text_side
    else:
        if volume.umbilicus is not None:
            fields["umbilicus"] = volume.umbilicus
            if volume.centroid is not None:  # a proxy for the umbilicus: informational only
                cm = chirality_map(p, volume.centroid)
                cc = cm[ok & np.isfinite(cm) & (np.abs(np.nan_to_num(cm)) > MIN_ABS_DET)]
                sources["centroid_informational"] = dict(
                    sign=int(np.sign(np.sum(np.sign(cc)))) if cc.size else 0,
                    agreement_with_own_sign=float(max((cc > 0).mean(), (cc < 0).mean())) if cc.size else math.nan)
        elif volume.centroid is not None:
            fields["centroid"] = volume.centroid
        gc, cond = normal_line_centre(np.where(ok[..., None], p, -1.0))
        sources["normal_line_centre"] = dict(cond_median=cond, valid=bool(gc is not None and cond <= COND_MAX))
        if gc is not None and cond <= COND_MAX:
            fields["normal_line_centre"] = gc
        elif gc is not None:
            flags.append(f"normal-line centre ill-conditioned (cond {cond:.1f} > {COND_MAX})")
    if not fields:
        return _flagged("FLAG", conv, flags + ["no valid r_out source"], sources)

    maps = {k: chirality_map(p, v) for k, v in fields.items()}
    clear = {k: ok & np.isfinite(m) & (np.abs(np.nan_to_num(m)) > MIN_ABS_DET) for k, m in maps.items()}
    signs = {}
    for k, m in maps.items():
        c = m[clear[k]]
        s = int(np.sign(np.sum(np.sign(c)))) if c.size else 0
        signs[k] = s
        sources.setdefault(k, {}).update(sign=s, n_clear=int(c.size))
    nonzero = {s for s in signs.values() if s != 0}
    if not nonzero:
        return _flagged("FLAG", conv, flags + ["chirality undetermined under every source"], sources)
    if len(nonzero) > 1:
        return _flagged("FLAG", conv, flags + [f"sources disagree on the whole-mesh sign {signs}: read both mirror states"],
                        sources)
    s = nonzero.pop()

    # reversed under EVERY source (fold / sheet jump) versus source disagreement
    all_clear = np.logical_and.reduce(list(clear.values()))
    rev = all_clear.copy()
    for m in maps.values():
        rev &= np.sign(np.nan_to_num(m)) == -s
    n_all = int(all_clear.sum())
    rev_frac = float(rev.sum() / n_all) if n_all else math.nan
    if n_all and rev_frac > REVERSED_MASK_MAX:
        return _flagged("FLAG", conv, flags + [f"reversed fraction {rev_frac:.3f} > {REVERSED_MASK_MAX}: read both mirror states"],
                        sources, rev_frac, s)
    rev_zone = binary_dilation(rev, iterations=1) if rev.any() else rev
    keep_out = ok & ~rev_zone
    if rev.any():
        flags.append(f"reversed patch masked out: {rev_frac:.3f} of clear vertices")

    for k, m in maps.items():
        c = m[clear[k] & ~rev_zone]
        sources[k]["agreement"] = float((np.sign(c) == s).mean()) if c.size else math.nan
    pair = {}
    ks = sorted(maps)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            both = clear[ks[i]] & clear[ks[j]] & ~rev_zone
            if both.any():
                pair[f"{ks[i]}_vs_{ks[j]}"] = float((np.sign(maps[ks[i]][both]) == np.sign(maps[ks[j]][both])).mean())
    sources["pairwise_agreement"] = pair
    ags = [sources[k]["agreement"] for k in maps]
    pv = list(pair.values())
    if len(maps) >= 2 and min(ags) >= 0.98 and min(pv or [1.0]) >= 0.95 and rev_frac <= REVERSED_CLEAN_MAX:
        conf = "HIGH"
    elif len(maps) >= 2 and min(ags) >= 0.9 and min(pv or [1.0]) >= 0.8:
        conf = "MEDIUM"
    elif len(maps) == 1 and ags[0] >= 0.95 and rev_frac <= REVERSED_CLEAN_MAX:
        conf = "MEDIUM"
        flags.append(f"single valid source ({ks[0]})")
    else:
        return _flagged("FLAG", conv, flags + [f"weak agreement {dict(zip(maps, ags))}, pairwise {pair}, "
                                               f"reversed {rev_frac:.3f} (HIGH and single-source MEDIUM need reversed <= {REVERSED_CLEAN_MAX})"],
                        sources, rev_frac, s, keep_out)

    # flips (never rotate): rows first, then columns
    z = np.where(keep_out, p[..., 2], np.nan)
    dzv = np.diff(z, axis=0)
    trend = float(np.nanmedian(dzv)) if np.isfinite(dzv).any() else 0.0
    if volume.z_top_to_bottom is None or trend == 0.0:
        flip_v = None
        if volume.z_top_to_bottom is None:
            flags.append("up/down unknown (z_direction_is_top_to_bottom null): 180 degree ambiguity only")
        else:
            flags.append("rows not graded in z: level the mesh first for a defined flip_v")
        mirror_u = bool(s != conv)
        r180 = True
    else:
        flip_v = bool((trend > 0) != volume.z_top_to_bottom)
        current = -s if flip_v else s
        mirror_u = bool(current != conv)
        r180 = False
    return HandedDecision(state="OK", confidence=conf, convention_sign=conv, chirality=s, mirror_u=mirror_u,
                          flip_v=flip_v, rot180_equivalent=r180, keep=keep_out, reversed_frac=rev_frac,
                          sources=sources, flags=flags)


# ---------------------------------------------------------------------------
# up/down evidence (F5): inward rotation sense along rows spanning > 1 turn
# ---------------------------------------------------------------------------

def winding_sense(points: np.ndarray, centre: CentreCurve, *, min_pairs: int = 20) -> dict:
    """w = +1 if moving along a row by one full turn in +phi (about +z)
    decreases the radius (inward is counter-clockwise), -1 if it increases,
    0 if no row spans a full turn. Rolls read inward (beginning outside), so
    text-top is at low z iff w * parity = +1."""
    p = np.asarray(points, np.float64)
    ok = valid_mask(p)
    cx, cy = centre(p[..., 2])
    r = np.hypot(p[..., 0] - cx, p[..., 1] - cy)
    phi = np.arctan2(p[..., 1] - cy, p[..., 0] - cx)
    drs, span_max = [], 0.0
    for i in range(p.shape[0]):
        cols = np.nonzero(ok[i])[0]
        if cols.size < 3:
            continue
        for run in np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1):
            if run.size < 3:
                continue
            ph = np.unwrap(phi[i, run])
            span_max = max(span_max, float(ph.max() - ph.min()))
            if ph.max() - ph.min() <= 2 * np.pi:
                continue
            o = np.argsort(ph)
            phs, rs = ph[o], r[i, run][o]
            tgt = phs + 2 * np.pi
            j = np.searchsorted(phs, tgt)
            m = j < phs.size
            j1 = j[m]
            j0 = np.clip(j1 - 1, 0, phs.size - 1)
            t = (tgt[m] - phs[j0]) / np.maximum(phs[j1] - phs[j0], 1e-12)
            drs.extend((rs[j0] + t * (rs[j1] - rs[j0]) - rs[m]).tolist())
    d = np.array(drs)
    out = dict(max_row_span_turns=span_max / (2 * np.pi), n_pairs=int(d.size))
    if d.size >= min_pairs:
        fneg = float((d < 0).mean())
        out.update(w=1 if fneg > 0.5 else -1, agreement=max(fneg, 1 - fneg), dr_per_turn_median=float(np.median(d)))
    else:
        out.update(w=0, agreement=math.nan, dr_per_turn_median=math.nan)
    return out


def up_from_winding(w: int, parity: int) -> bool | None:
    """Text top at low z (True) / high z (False) predicted by F5; None if w or parity is 0."""
    if w == 0 or parity == 0:
        return None
    return bool(w * parity == 1)


CentreLike = Union[CentreCurve, Callable]
