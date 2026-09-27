"""Reproduce results/official_meshes.{csv,json}: the rule on every organizer
tifxyz mesh of PHerc0139, PHerc0009B, PHerc0800 and PHerc1447 in the listed
volume frame, from public data only (catalogue, meshes, official umbilicus,
level-4 masked CT for the centroid track and the frame check). Each mesh is
also decided without the frame gate (columns *_without_frame_gate), to show
what the gate changes.

    python scripts/reproduce_official.py CACHE_DIR results/ [SCROLL ...]

Downloads about 0.6 GB of meshes and some level-4 CT planes (cached in
CACHE_DIR). Give scroll ids (e.g. PHerc0800) to run only those.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from handcheck import public as P
from handcheck.cli import CSV_FIELDS, decide
from handcheck.rule import VolumeInfo, frame_check_from_volume
from handcheck.tifxyz import read_tifxyz

# (scroll, volume id of the organizer meshes); meshes on other volumes of the same scroll are not included
TARGETS = [("PHerc0139", "20250728140407"), ("PHerc0009B", "20250521125136"),
           ("PHerc0800", "20250521135224"), ("PHerc1447", "20250521151220")]
LEVEL, ZSTRIDE, FRAME_STRIDE = 4, 8, 4


def main(cache: Path, outdir: Path, only: list[str]) -> None:
    cat = P.catalogue(cache / "metadata.json")
    rows, prov = [], {}
    for scroll, vol in [t for t in TARGETS if not only or t[0] in only]:
        t0 = time.time()
        ve = P.volume_entry(cat, scroll, vol)
        props = ve["properties"]
        prefix = P.volume_prefix(cat, scroll, vol)
        zattrs = json.loads(P._get(f"{prefix}/.zattrs"))
        stype = cat["samples"][scroll]["sample"]["properties"].get("type") or "scroll"
        z4 = P.ZarrLevel(prefix, LEVEL)
        umb, uinfo = (P.umbilicus(cat, scroll, vol) if stype == "scroll" else (None, {}))
        cen, cinfo = None, {}
        if stype == "scroll" and umb is None:
            f = cache / "centroids" / f"{scroll}-{vol}-L{LEVEL}-z{ZSTRIDE}.json"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(P.centroid_track(z4, ZSTRIDE)))
            cen, cinfo = P.centroid_curve(json.loads(f.read_text()))
        vi = VolumeInfo.from_metadata(zattrs, props, sample_type=stype, umbilicus=umb, centroid=cen,
                                      name=f"{scroll}/{vol}")
        meshes = P.fetch_meshes(scroll, vol, cache / "meshes" / scroll, cat)
        prov[scroll] = dict(volume=vol, volume_long_id=ve["long_id"], voxel_um=props.get("pixel_size_um"),
                            sample_type=stype, parity_inputs=dict(axes=[a["name"] for a in zattrs["multiscales"][0]["axes"]],
                                                                  left_handed_coordinates=props.get("left_handed_coordinates")),
                            z_direction_is_top_to_bottom=props.get("z_direction_is_top_to_bottom"),
                            text_side=props.get("text_side"), umbilicus=uinfo, centroid=cinfo, n_meshes=len(meshes))
        for m in meshes:
            fr = frame_check_from_volume(read_tifxyz(m).points, z4.sample, z4.shape_level0, stride=FRAME_STRIDE)
            r = decide(m, vi, fr)
            nf = decide(m, vi, None)  # the same decision without the frame gate, for comparison only
            r["without_frame_gate"] = {k: nf[k] for k in ("state", "confidence", "mirrored_as_stored")}
            r.update(scroll=scroll, segment=m.parent.name, volume=vol, voxel_um=props.get("pixel_size_um"))
            rows.append(r)
            print(scroll, m.parent.name[:40], r["state"], r["confidence"], r["mirrored_as_stored"],
                  f"zero {fr.frac_zero_ct:.3f} out {fr.frac_outside:.3f}", flush=True)
        prov[scroll]["seconds"] = round(time.time() - t0, 1)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "official_meshes.json").write_text(json.dumps(dict(inputs=prov, meshes=rows), indent=1))
    fields = (["scroll", "segment", "volume", "voxel_um"] + [f for f in CSV_FIELDS if f != "mesh"]
              + ["state_without_frame_gate", "confidence_without_frame_gate", "mirrored_without_frame_gate", "mesh"])
    with open(outdir / "official_meshes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            fr = r["frame"] or {}
            nf = r["without_frame_gate"]
            w.writerow({**r, "frac_zero_ct": fr.get("frac_zero_ct"), "frac_outside": fr.get("frac_outside"),
                        "state_without_frame_gate": nf["state"], "confidence_without_frame_gate": nf["confidence"],
                        "mirrored_without_frame_gate": nf["mirrored_as_stored"],
                        "sources_used": "+".join(r["sources_used"]), "flags": "; ".join(r["flags"])})
    by = {}
    for r in rows:
        k = (r["scroll"], r["state"], r["confidence"], r["mirrored_as_stored"])
        by[k] = by.get(k, 0) + 1
    for k, v in sorted(by.items(), key=str):
        print(v, *k)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
