"""Independent up/down check (F5) on meshes that span more than one turn:
rolls read from the outside inward, so the rotation sense of the inward
direction about +z (w), times the frame parity, predicts whether the text top
is at low z. Compared with the catalogue flag z_direction_is_top_to_bottom.
Uses the official umbilicus as centre. Numbers only.

    python scripts/winding_check.py CACHE_DIR results/winding_PHerc0139.json [PHerc0139 20250728140407]
"""

import json
import sys
from pathlib import Path

import numpy as np

from handcheck import public as P
from handcheck.rule import frame_parity, up_from_winding, winding_sense
from handcheck.tifxyz import read_tifxyz


def best(points, centre):
    """Winding sense along rows, or along columns if those span more turns."""
    a = winding_sense(points, centre)
    b = winding_sense(np.ascontiguousarray(np.swapaxes(points, 0, 1)), centre)
    a["axis"], b["axis"] = "rows", "columns"
    return a if a["max_row_span_turns"] >= b["max_row_span_turns"] else b


def main(cache: Path, out: Path, scroll: str = "PHerc0139", vol: str = "20250728140407") -> None:
    cat = P.catalogue(cache / "metadata.json")
    props = P.volume_entry(cat, scroll, vol)["properties"]
    zattrs = json.loads(P._get(f"{P.volume_prefix(cat, scroll, vol)}/.zattrs"))
    parity, _ = frame_parity(zattrs, props)
    umb, info = P.umbilicus(cat, scroll, vol)
    flag = props.get("z_direction_is_top_to_bottom")
    rows = []
    for m in P.fetch_meshes(scroll, vol, cache / "meshes" / scroll, cat):
        w = best(read_tifxyz(m).points.astype(np.float64), umb)
        pred = up_from_winding(w["w"], parity)
        rows.append(dict(segment=m.parent.name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.items()},
                         predicted_top_at_low_z=pred, consistent=(None if pred is None else pred == flag)))
        print(m.parent.name[:40], w["w"], round(w["max_row_span_turns"], 2), pred, rows[-1]["consistent"], flush=True)
    n_test = sum(r["consistent"] is not None for r in rows)
    n_ok = sum(r["consistent"] is True for r in rows)
    summary = dict(scroll=scroll, volume=vol, parity=parity, z_direction_is_top_to_bottom=flag, umbilicus=info,
                   n_meshes=len(rows), n_spanning_one_turn=n_test, n_consistent=n_ok)
    print(json.dumps(summary))
    out.write_text(json.dumps(dict(summary=summary, meshes=rows), indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
