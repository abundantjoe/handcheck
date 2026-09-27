"""Planted-glyph validation on synthetic spiral scrolls (numbers only).

For each glyph (R, K), frame (ascending, rot180x, reflect_z, reflect_x) and
input parameterisation (as built, u-mirrored, v-flipped, both) = 32 cases:
level the mesh, decide the orientation with the rule, render along normals,
and correlate (cv2 TM_CCOEFF_NORMED) with the glyph and its mirror images.
The wrong-sign control applies the opposite mirror decision to the same case.

    python scripts/synthetic_validation.py results/synthetic_validation.json
"""

import dataclasses
import json
import sys

import cv2
import numpy as np

from handcheck import level_mesh as L
from handcheck import render_synth as RS
from handcheck import rule as HR
from handcheck import synth as S
from handcheck import tifxyz as T

SHAPE, CENTRE, PIX, Z_C, PHI_G, K_G, MARGIN = (110, 140, 140), (70.0, 70.0), 1.5, 55.0, np.radians(35.0), 1, 0.2


def ncc(img, tpl):
    return float(cv2.matchTemplate(np.nan_to_num(np.asarray(img, np.float32)), tpl.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED).max())


def main(out: str) -> None:
    sp = S.Spiral(centre=CENTRE, r_outer=62.0, pitch=9.0, turns=4.5, phi_start=np.radians(-20.0), w=1)
    n, a = 44, np.radians(30.0)
    uc = (n - 1) / 2
    s0 = -PIX * (uc * np.cos(a) - uc * np.sin(a))
    z0 = Z_C - PIX * (uc * np.sin(a) + uc * np.cos(a))
    base = S.spiral_sheet_grid(sp, K_G, (n, n), PIX, 30.0, s0=s0, z0=z0, phi_ref=PHI_G)
    cases = []
    for ch in ("R", "K"):
        g = S.font_glyph(ch)
        sc = S.spiral_glyph_volume(SHAPE, sp, g, k_glyph=K_G, phi_g=PHI_G, z_c=Z_C, pix=PIX)
        for kind in ("ascending", "rot180x", "reflect_z", "reflect_x"):
            for pflip in ("none", "u", "v", "uv"):
                src = base
                if "u" in pflip:
                    src = np.ascontiguousarray(src[:, ::-1])
                if "v" in pflip:
                    src = np.ascontiguousarray(src[::-1])
                vol, pts, props = S.transform_scene(sc.vol, src, kind)
                lv = L.level_mesh(pts, step_vox=PIX)
                vi = HR.VolumeInfo.from_metadata(S.ZATTRS_ZYX, props, centroid=S.centroid_curve_from_volume(vol))
                d = HR.orient_for_reading(lv.points, vi, frame=HR.FrameCheck(0.0, 0.0))
                row = dict(glyph=ch, frame=kind, input=pflip, state=d.state, confidence=d.confidence)
                for tag, dec in (("rule", d), ("wrong_sign", dataclasses.replace(d, mirror_u=not d.mirror_u))):
                    q = dec.apply(lv.points)
                    img = RS.render_along_normals(vol, q, T.vertex_normals(q), n=5).mean(0)
                    row[tag] = dict(id=ncc(img, g), lr=ncc(img, np.fliplr(g)))
                cases.append(row)
    ok = [c for c in cases if c["rule"]["id"] > 0.8 and c["rule"]["id"] > c["rule"]["lr"] + MARGIN]
    ctl = [c for c in cases if c["wrong_sign"]["lr"] > 0.8 and c["wrong_sign"]["lr"] > c["wrong_sign"]["id"] + MARGIN]
    summary = dict(n_cases=len(cases), rule_unmirrored=len(ok), wrong_sign_mirrored=len(ctl),
                   rule_min_ncc=min(c["rule"]["id"] for c in cases),
                   rule_min_margin_over_mirror=min(c["rule"]["id"] - c["rule"]["lr"] for c in cases),
                   wrong_sign_min_margin=min(c["wrong_sign"]["lr"] - c["wrong_sign"]["id"] for c in cases),
                   pass_bar=dict(ncc=0.8, margin=MARGIN))
    print(json.dumps(summary, indent=1))
    with open(out, "w") as f:
        json.dump(dict(summary=summary, cases=cases), f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_validation.json")
