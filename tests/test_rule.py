"""handcheck.rule on synthetic scrolls (numbers only: renders
are scored by normalised cross-correlation against a font-rendered glyph and
its mirror images; nothing is displayed)."""

import cv2
import numpy as np
import pytest

from handcheck import rule as HR
from handcheck import synth as S
from handcheck import level_mesh as L
from handcheck import render_synth as RS
from handcheck import tifxyz as T

SHAPE = (110, 140, 140)
CENTRE = (70.0, 70.0)
PIX = 1.5
Z_C = 55.0
PHI_G = np.radians(35.0)
K_G = 1
MARGIN = 0.2
ZYX = S.ZATTRS_ZYX
FRAME_OK = HR.FrameCheck(0.0, 0.0)


def spiral(w=1):
    return S.Spiral(centre=CENTRE, r_outer=62.0, pitch=9.0, turns=4.5, phi_start=np.radians(-20.0), w=w)


def _score(img, tpl):
    return float(cv2.matchTemplate(np.nan_to_num(np.asarray(img, np.float32)), tpl.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED).max())


def scores(img, g):
    return dict(id=_score(img, g), lr=_score(img, np.fliplr(g)), ud=_score(img, np.flipud(g)),
                rot180=_score(img, np.rot90(g, 2)))


@pytest.fixture(scope="module", params=["R", "K"])
def scene(request):
    g = S.font_glyph(request.param)
    sc = S.spiral_glyph_volume(SHAPE, spiral(), g, k_glyph=K_G, phi_g=PHI_G, z_c=Z_C, pix=PIX)
    sc.char = request.param
    return sc


def source_grid(sc, angle_deg=30.0, n=44):
    a = np.radians(angle_deg)
    uc = vc = (n - 1) / 2
    s0 = -PIX * (uc * np.cos(a) - vc * np.sin(a))
    z0 = Z_C - PIX * (uc * np.sin(a) + vc * np.cos(a))
    return S.spiral_sheet_grid(sc.spiral, K_G, (n, n), PIX, angle_deg, s0=s0, z0=z0, phi_ref=PHI_G)


def volume_info(vol, props, **kw):
    return HR.VolumeInfo.from_metadata(ZYX, props, centroid=S.centroid_curve_from_volume(vol), **kw)


def read(vol, pts, d, flip_v=None):
    lvp = d.apply(pts, flip_v=flip_v)
    return RS.render_along_normals(vol, lvp, T.vertex_normals(lvp), n=5).mean(0)


# ---------------------------------------------------------------------------
# the rule renders the planted glyph unmirrored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ascending", "rot180x", "reflect_z", "reflect_x"])
@pytest.mark.parametrize("pflip", ["none", "u", "v", "uv"])
def test_planted_glyph_unmirrored(scene, kind, pflip):
    src = source_grid(scene)
    if "u" in pflip:
        src = np.ascontiguousarray(src[:, ::-1])
    if "v" in pflip:
        src = np.ascontiguousarray(src[::-1])
    vol, pts, props = S.transform_scene(scene.vol, src, kind)
    lv = L.level_mesh(pts, step_vox=PIX)
    d = HR.orient_for_reading(lv.points, volume_info(vol, props), frame=FRAME_OK)
    assert d.state == "OK" and d.confidence in ("HIGH", "MEDIUM"), d.flags
    assert d.convention_sign == (-1 if kind.startswith("reflect") else 1)
    s = scores(read(vol, lv.points, d), scene.glyph)
    assert s["id"] > 0.8 and s["id"] > s["lr"] + MARGIN, s
    if scene.char == "R":  # K is nearly up-down symmetric
        assert s["id"] > s["ud"] + MARGIN and s["id"] > s["rot180"] + MARGIN, s


def test_decision_fields_on_native_grid(scene):
    """Organizer-style grid (u ~ +phi, v ~ +z), ascending, parity +1: legible as stored."""
    src = source_grid(scene, angle_deg=0.0)
    d = HR.orient_for_reading(src, volume_info(scene.vol, dict(left_handed_coordinates=False,
                                                              z_direction_is_top_to_bottom=True)), frame=FRAME_OK)
    assert d.chirality == 1 and d.mirror_u is False and d.flip_v is False and not d.rot180_equivalent
    assert set(k for k in d.sources if k != "pairwise_agreement") >= {"centroid", "normal_line_centre"}
    assert d.confidence == "HIGH" and d.readable


def test_updown_unknown_gives_both_upright_candidates(scene):
    if scene.char != "R":
        pytest.skip("K is up-down symmetric: its 180 deg rotation equals its mirror image")
    src = source_grid(scene)
    lv = L.level_mesh(src, step_vox=PIX)
    d = HR.orient_for_reading(lv.points, volume_info(scene.vol, dict(left_handed_coordinates=False,
                                                                    z_direction_is_top_to_bottom=None)), frame=FRAME_OK)
    assert d.flip_v is None and d.rot180_equivalent
    for fv in (False, True):  # both choices are unmirrored; one is upright, the other rotated 180 deg
        s = scores(read(scene.vol, lv.points, d, flip_v=fv), scene.glyph)
        assert max(s["id"], s["rot180"]) > s["lr"] + MARGIN, (fv, s)
    with pytest.raises(ValueError):
        HR.orient_for_reading(lv.points, HR.VolumeInfo(parity=0, z_top_to_bottom=True), frame=FRAME_OK).apply(lv.points)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_parity_unknown_is_flagged(scene):
    vi = HR.VolumeInfo.from_metadata(ZYX, dict(z_direction_is_top_to_bottom=True))
    d = HR.orient_for_reading(source_grid(scene), vi, frame=FRAME_OK)
    assert d.state == "PARITY_UNKNOWN" and d.confidence == "FLAG" and d.mirror_u is None
    bad_axes = {"multiscales": [{"axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}]}]}
    assert HR.frame_parity(bad_axes, dict(left_handed_coordinates=False))[0] == 0


def test_frame_gate(scene):
    vi = volume_info(scene.vol, dict(left_handed_coordinates=False, z_direction_is_top_to_bottom=True))
    src = source_grid(scene)
    assert HR.orient_for_reading(src, vi, frame=HR.FrameCheck(0.036, 0.0)).state == "OK"  # 0139 worst control
    for fc in (HR.FrameCheck(0.41, 0.0), HR.FrameCheck(0.0, 0.05)):  # 1447 frame-bad group
        d = HR.orient_for_reading(src, vi, frame=fc)
        assert d.state == "FRAME_UNVERIFIED" and d.mirror_u is None
    assert "frame not checked" in HR.orient_for_reading(src, vi).flags


def test_frame_check_from_volume(scene):
    vol = scene.vol
    samp = lambda P: vol[P[:, 2].astype(int), P[:, 1].astype(int), P[:, 0].astype(int)]  # noqa: E731
    src = source_grid(scene)
    fc = HR.frame_check_from_volume(src, samp, vol.shape, stride=1)
    assert fc.ok and fc.frac_outside == 0.0
    shifted = src.copy()
    shifted[..., 2] += 200.0  # mis-mapped frame: above the volume
    fc2 = HR.frame_check_from_volume(shifted, samp, vol.shape, stride=1)
    assert not fc2.ok and fc2.frac_outside == 1.0


def _sheet_jump_grid(sp, frac_cols):
    """Winding-2 patch whose last columns jump to winding 3 traversed
    BACKWARD in u: an orientation-reversed patch (like a mis-traced sheet jump)."""
    n = 40
    a = S.spiral_sheet_grid(sp, 2, (n, n), 2.0, 0.0, s0=-40.0, z0=20.0, phi_ref=np.radians(60.0))
    m = int(round(frac_cols * n))
    if m:
        b = S.spiral_sheet_grid(sp, 3, (n, m), 2.0, 0.0, s0=-40.0 + 2.0 * (n - m), z0=20.0, phi_ref=np.radians(60.0))
        a[:, n - m:] = b[:, ::-1]
    return a


def test_reversed_patch_masked(scene):
    vi = volume_info(scene.vol, dict(left_handed_coordinates=False, z_direction_is_top_to_bottom=True))
    g = _sheet_jump_grid(spiral(), 0.1)
    d = HR.orient_for_reading(g, vi, frame=FRAME_OK)
    assert d.state == "OK" and 0.05 < d.reversed_frac <= 0.2, (d.reversed_frac, d.flags)
    assert not d.keep[:, -2:].any() and d.keep[:, :20].all()
    assert any("reversed patch masked" in f for f in d.flags)
    d2 = HR.orient_for_reading(_sheet_jump_grid(spiral(), 0.3), vi, frame=FRAME_OK)
    assert d2.confidence == "FLAG" and d2.reversed_frac > 0.2


def test_ill_conditioned_normal_line_centre_is_not_a_source():
    """A flat sheet has parallel normals: its normal-line centre is undefined
    and must not vote; with a centroid present the decision is single-source."""
    vv, uu = np.mgrid[0:30, 0:30].astype(np.float64)
    p = np.stack([400.0 + 5 * uu, np.full_like(uu, 900.0), 100 + 5 * vv], -1)  # plane y = 900, du x dv = -y
    cc = L.CentreCurve.straight(475.0, 400.0)  # centre on the -y side: r_out = +y -> chirality -1
    vi = HR.VolumeInfo(parity=1, z_top_to_bottom=True, centroid=cc)
    d = HR.orient_for_reading(p, vi, frame=FRAME_OK)
    assert d.sources["normal_line_centre"]["valid"] is False
    assert HR.normal_line_centre(p) == (None, float("inf"))
    assert d.confidence == "MEDIUM" and d.chirality == -1 and d.mirror_u is True
    assert any("single valid source" in f for f in d.flags)


def test_source_disagreement_is_flagged(scene):
    """A wrong centroid (on the convex side of the sheet) against a valid
    mesh centre: whole-mesh signs disagree -> FLAG, read both mirror states."""
    src = source_grid(scene)
    bad = L.CentreCurve.straight(CENTRE[0] + 200 * np.cos(PHI_G), CENTRE[1] + 200 * np.sin(PHI_G))
    d = HR.orient_for_reading(src, HR.VolumeInfo(parity=1, z_top_to_bottom=True, centroid=bad), frame=FRAME_OK)
    assert d.confidence == "FLAG" and any("disagree" in f for f in d.flags)


def test_keep_mask_excludes_vertices(scene):
    vi = volume_info(scene.vol, dict(left_handed_coordinates=False, z_direction_is_top_to_bottom=True))
    src = source_grid(scene)
    keep = np.zeros(src.shape[:2], bool)
    keep[:, :20] = True
    d = HR.orient_for_reading(src, vi, frame=FRAME_OK, keep=keep)
    assert d.state == "OK" and not d.keep[:, 20:].any()


# ---------------------------------------------------------------------------
# fragments
# ---------------------------------------------------------------------------

def test_fragment_text_side():
    g = S.font_glyph("R")
    Z, Y, X = 70, 40, 70
    zz, yy, xx = np.meshgrid(np.arange(Z, dtype=float), np.arange(Y, dtype=float), np.arange(X, dtype=float),
                             indexing="ij")
    col = (35.0 - xx) / PIX + (g.shape[1] - 1) / 2  # viewer on -y, top = low z: right = -x
    row = (zz - 35.0) / PIX + (g.shape[0] - 1) / 2
    ink = cv2.remap(g, col[:, 0, :].astype(np.float32), row[:, 0, :].astype(np.float32), cv2.INTER_LINEAR)
    vol = (0.3 * np.exp(-0.5 * ((yy - 20.0) / 1.0) ** 2)
           + ink[:, None, :] * np.exp(-0.5 * ((yy - 18.8) / 0.9) ** 2)).astype(np.float32)
    vv, uu = np.mgrid[0:40, 0:40].astype(float)
    src = np.stack([5 + PIX * uu, np.full_like(uu, 20.0), 5 + PIX * vv], -1)
    vi = HR.VolumeInfo.from_metadata(ZYX, dict(left_handed_coordinates=False, z_direction_is_top_to_bottom=True,
                                               text_side="-y"), sample_type="fragment")
    for s in (src, np.ascontiguousarray(src[:, ::-1])):
        d = HR.orient_for_reading(s, vi, frame=FRAME_OK)
        assert d.confidence == "MEDIUM" and d.state == "OK"
        sc = scores(read(vol, s, d), g)
        assert sc["id"] > 0.8 and sc["id"] > sc["lr"] + MARGIN, sc
    no_side = HR.VolumeInfo.from_metadata(ZYX, dict(left_handed_coordinates=False), sample_type="fragment")
    assert HR.orient_for_reading(src, no_side, frame=FRAME_OK).confidence == "FLAG"


# ---------------------------------------------------------------------------
# up/down evidence (F5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ascending", "rot180x", "reflect_z", "reflect_x"])
def test_winding_predicts_up(kind):
    sp = spiral(1)  # physical roll: reading (inward) = counter-clockwise, top at low z when built
    g = S.spiral_row_grid(sp, 400, 6, 10.0, 5.0, theta0=0.3, dtheta=2.5 * 2 * np.pi / 400)
    _, g2, props = S.transform_scene(np.zeros(SHAPE, np.float32), g, kind)
    Z, Y, X = SHAPE
    c = [CENTRE[0], CENTRE[1]]
    if kind == "rot180x":
        c[1] = (Y - 1) - c[1]
    if kind == "reflect_x":
        c[0] = (X - 1) - c[0]
    ws = HR.winding_sense(g2, L.CentreCurve.straight(*c))
    parity, _ = HR.frame_parity(ZYX, props)
    assert HR.up_from_winding(ws["w"], parity) == props["z_direction_is_top_to_bottom"]
    assert HR.winding_sense(g2[:, :100], L.CentreCurve.straight(*c))["w"] == 0  # < 1 turn
    assert HR.up_from_winding(0, 1) is None


def test_umbilicus_outranks_centroid(scene):
    """With an official umbilicus the centroid (its proxy) never votes: a poor
    centroid cannot veto the umbilicus and the mesh centre."""
    src = source_grid(scene)
    good = L.CentreCurve.straight(*CENTRE)
    bad = L.CentreCurve.straight(CENTRE[0] + 200 * np.cos(PHI_G), CENTRE[1] + 200 * np.sin(PHI_G))
    d = HR.orient_for_reading(src, HR.VolumeInfo(parity=1, z_top_to_bottom=True, umbilicus=good, centroid=bad),
                              frame=FRAME_OK)
    assert d.confidence == "HIGH" and "centroid" not in d.sources
    assert d.sources["centroid_informational"]["sign"] == -1


# ---------------------------------------------------------------------------
# wrong-sign control: the opposite mirror decision must render the glyph mirrored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ascending", "rot180x", "reflect_z", "reflect_x"])
@pytest.mark.parametrize("pflip", ["none", "u", "v", "uv"])
def test_wrong_sign_control_is_mirrored(scene, kind, pflip):
    import dataclasses

    src = source_grid(scene)
    if "u" in pflip:
        src = np.ascontiguousarray(src[:, ::-1])
    if "v" in pflip:
        src = np.ascontiguousarray(src[::-1])
    vol, pts, props = S.transform_scene(scene.vol, src, kind)
    lv = L.level_mesh(pts, step_vox=PIX)
    d = HR.orient_for_reading(lv.points, volume_info(vol, props), frame=FRAME_OK)
    wrong = dataclasses.replace(d, mirror_u=not d.mirror_u)
    s = scores(read(vol, lv.points, wrong), scene.glyph)
    assert s["lr"] > 0.8 and s["lr"] > s["id"] + MARGIN, s
