"""Command line: predict, for every tifxyz mesh in a folder, whether a render
of the grid as stored (u to the right, v downward) reads mirrored.

    handcheck MESH_OR_DIR [...] --props props.json [--axes zyx | --zattrs .zattrs]
              [--umbilicus umb.json] [--centroid centre.json]
              [--frame frame.json | --ct-zarr VOLUME.zarr --ct-level 4]
              [--fragment --text-side -y] [--json out.json] [--csv out.csv] [--strict]

Exit codes: 0 = every mesh processed; 1 = runtime error; 2 = usage error;
3 = ``--strict`` and at least one mesh is not decided (state != OK).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .level_mesh import CentreCurve
from .rule import FrameCheck, VolumeInfo, frame_check_from_volume, orient_for_reading
from .tifxyz import read_tifxyz

PUBLIC = "s3://vesuvius-challenge-open-data/"
CSV_FIELDS = ["mesh", "state", "confidence", "mirrored_as_stored", "mirror_u", "flip_v", "rot180_equivalent",
              "parity", "chirality", "reversed_frac", "frac_zero_ct", "frac_outside", "sources_used", "flags"]


def find_meshes(paths: list[str]) -> list[Path]:
    """Every tifxyz directory (holding x.tif, y.tif, z.tif, meta.json) at or below the given paths."""
    out: list[Path] = []
    for p in map(Path, paths):
        if (p / "x.tif").exists():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(d.parent for d in p.rglob("x.tif") if (d.parent / "meta.json").exists()))
        else:
            raise FileNotFoundError(f"not a tifxyz directory or folder: {p}")
    return sorted(set(out))


def load_centre(path: str | None) -> CentreCurve | None:
    """Centre curve from JSON: a villa umbilicus file ({"control_points": [{"x","y","z"}, ...]}),
    a list of [x, y, z] points, or {"z": [...], "x": [...], "y": [...]}. Level-0 voxel units."""
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "control_points" in d:
        P = np.array([[c["x"], c["y"], c["z"]] for c in d["control_points"]], float)
    elif isinstance(d, dict) and {"x", "y", "z"} <= set(d):
        return CentreCurve(z=np.asarray(d["z"], float), x=np.asarray(d["x"], float), y=np.asarray(d["y"], float))
    else:
        P = np.asarray(d, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) == 0:
        raise ValueError(f"cannot read a centre curve from {path}")
    return CentreCurve(z=P[:, 2], x=P[:, 0], y=P[:, 1])


def zattrs_for(axes: str | None, zattrs: str | None) -> dict | None:
    if zattrs:
        return json.loads(Path(zattrs).read_text())
    if axes:
        return {"multiscales": [{"axes": [{"name": a} for a in axes.lower()]}]}
    return None


def zarr_sampler(path: str, level: int):
    """(sampler, level-0 shape zyx) for nearest-voxel sampling of an OME-Zarr level. Needs ``zarr``."""
    try:
        import zarr
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit("--ct-zarr needs the optional dependency: pip install zarr") from e
    arr = zarr.open_array(f"{path.rstrip('/')}/{level}", mode="r")
    f = 2 ** level
    Z, Y, X = arr.shape

    def sample(P: np.ndarray) -> np.ndarray:
        idx = np.floor(P / f).astype(np.int64)
        xi, yi, zi = (np.clip(idx[:, 0], 0, X - 1), np.clip(idx[:, 1], 0, Y - 1), np.clip(idx[:, 2], 0, Z - 1))
        return np.asarray(arr.vindex[zi, yi, xi])

    return sample, (Z * f, Y * f, X * f)


def _num(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, dict):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    return x


def decide(mesh_dir: Path, volume: VolumeInfo, frame: FrameCheck | None) -> dict:
    p = read_tifxyz(mesh_dir).points
    d = orient_for_reading(p, volume, frame=frame)
    mirrored = None if d.state != "OK" else bool(d.chirality != d.convention_sign)
    used = sorted(k for k, v in d.sources.items()
                  if isinstance(v, dict) and "sign" in v and k != "centroid_informational")
    return _num(dict(mesh=mesh_dir.name, state=d.state, confidence=d.confidence, mirrored_as_stored=mirrored,
                     mirror_u=d.mirror_u, flip_v=d.flip_v, rot180_equivalent=d.rot180_equivalent,
                     parity=d.convention_sign, chirality=d.chirality, reversed_frac=d.reversed_frac,
                     frame=(vars(frame) if frame is not None else None), sources_used=used,
                     sources=d.sources, flags=d.flags))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="handcheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="tifxyz directories, or folders searched recursively for them")
    ap.add_argument("--props", required=True,
                    help="JSON with the volume's catalogue properties: left_handed_coordinates, "
                         "z_direction_is_top_to_bottom, and text_side for fragments")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--axes", help="volume array axis names in storage order, e.g. zyx")
    g.add_argument("--zattrs", help="the volume's OME-Zarr .zattrs (axis names are read from multiscales[0].axes)")
    ap.add_argument("--umbilicus", help="official umbilicus JSON on the same volume (level-0 voxels)")
    ap.add_argument("--centroid", help="CT-mask centroid curve JSON on the same volume (level-0 voxels)")
    f = ap.add_mutually_exclusive_group()
    f.add_argument("--frame", help="JSON {mesh_name: {frac_zero_ct, frac_outside}} from a previous frame check")
    f.add_argument("--ct-zarr", help="masked OME-Zarr volume for the frame check: a local path (needs zarr) or "
                                     "s3://vesuvius-challenge-open-data/<scroll>/volumes/<volume>.zarr (read over HTTPS)")
    ap.add_argument("--ct-level", type=int, default=4, help="pyramid level for --ct-zarr (default 4)")
    ap.add_argument("--fragment", action="store_true", help="detached fragment: use -text_side instead of a centre")
    ap.add_argument("--text-side", help="override text_side for --fragment, e.g. -y")
    ap.add_argument("--json", help="write all decisions (full detail) to this JSON file")
    ap.add_argument("--csv", help="write a one-row-per-mesh CSV summary")
    ap.add_argument("--strict", action="store_true", help="exit 3 if any mesh is not decided")
    ap.add_argument("--version", action="version", version=f"handcheck {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        props = json.loads(Path(a.props).read_text())
        if a.text_side:
            props["text_side"] = a.text_side
        volume = VolumeInfo.from_metadata(zattrs_for(a.axes, a.zattrs), props,
                                          sample_type="fragment" if a.fragment else "scroll",
                                          umbilicus=load_centre(a.umbilicus), centroid=load_centre(a.centroid))
        frames = {k: FrameCheck(v["frac_zero_ct"], v["frac_outside"])
                  for k, v in (json.loads(Path(a.frame).read_text()) if a.frame else {}).items()}
        sampler = None
        if a.ct_zarr and a.ct_zarr.startswith(PUBLIC):
            from .public import ZarrLevel

            zl = ZarrLevel(a.ct_zarr[len(PUBLIC):], a.ct_level)
            sampler = (zl.sample, zl.shape_level0)
        elif a.ct_zarr:
            sampler = zarr_sampler(a.ct_zarr, a.ct_level)
        rows = []
        for m in find_meshes(a.paths):
            fr = frames.get(m.name)
            if sampler is not None:
                fr = frame_check_from_volume(read_tifxyz(m).points, sampler[0], sampler[1])
            r = decide(m, volume, fr)
            rows.append(r)
            ms = {None: "undecided", True: "MIRRORED", False: "unmirrored"}[r["mirrored_as_stored"]]
            print(f"{r['mesh']}\t{r['state']}\t{r['confidence']}\t{ms}\t{'; '.join(r['flags'])}")
    except (OSError, ValueError, KeyError) as e:
        print(f"handcheck: error: {e}", file=sys.stderr)
        return 1
    if not rows:
        print("handcheck: no tifxyz meshes found", file=sys.stderr)
        return 1
    if a.json:
        Path(a.json).write_text(json.dumps(dict(version=__version__, meshes=rows), indent=1))
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                fr = r["frame"] or {}
                w.writerow({**r, "frac_zero_ct": fr.get("frac_zero_ct"), "frac_outside": fr.get("frac_outside"),
                            "sources_used": "+".join(r["sources_used"]), "flags": "; ".join(r["flags"])})
    n_ok = sum(r["state"] == "OK" for r in rows)
    n_mir = sum(r["mirrored_as_stored"] is True for r in rows)
    print(f"# {len(rows)} meshes: {n_ok} decided ({n_mir} mirrored as stored), {len(rows) - n_ok} flagged",
          file=sys.stderr)
    return 3 if a.strict and n_ok < len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
