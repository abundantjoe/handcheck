# handcheck

Predicts whether a tifxyz mesh render reads mirrored, from geometry and catalogue metadata; flags undecidable meshes.

## Why it matters

A virtually unrolled segment can only be read if its render shows the written face (the recto) from the correct side. Otherwise every letter comes out as its mirror image. Two independent signs decide this:

- the **chirality** of the mesh parameterisation, `sign(det[dP/du, dP/dv, r_out])`, where `r_out` points away from the scroll's centre;
- the **parity** of the volume's index frame (`left_handed_coordinates` in the official catalogue, together with the zarr axis names).

The tracer, a manual u-flip, a reversed slice order or an affine between volumes can each change one of these signs without anyone noticing. Without a geometric check, the mirror state can only be judged by rendering and looking. On a new scroll, with no known text, looking does not settle it, and a mirrored render can go on to ink detection and reading unnoticed.

**Compared with existing tools:**
- The official renderer (`vc_render_tifxyz`) assumes the standard segment orientation. It offers `--flip` and `--rotate` options, but leaves the choice to the user.
- We are not aware of an existing tool that predicts the mirror state per mesh before rendering, or that checks whether a mesh lies on the scroll in the frame it declares.

`handcheck` gives the answer before any rendering, from the mesh and the catalogue. It also reports **why** it cannot decide when that is the case:
- the frame is unknown;
- the mesh is not on the scroll in the volume it declares;
- the independent centre estimates disagree;
- part of the mesh is orientation-reversed, which happens at folds and sheet jumps.

This addresses the Progress Prize requirement to "address a specific challenge using Vesuvius Challenge scroll data". It also fits the prize page's call to reveal "insightful, actionable information ... detecting failure-cases of existing methods on real scroll data". For one such failure case, see [Findings](#findings-on-real-data).

## Install

Python >= 3.10. Tested on Linux (Python 3.11, numpy 2.4, scipy 1.17, tifffile 2026.3).

```bash
pip install "git+https://github.com/abundantjoe/handcheck"          # runtime: numpy, scipy, tifffile
pip install "handcheck[test] @ git+https://github.com/abundantjoe/handcheck"   # + pytest, opencv (tests only)
pip install zarr                                                     # optional: frame check on a local OME-Zarr
```

## Quick start (public data)

```bash
git clone https://github.com/abundantjoe/handcheck && cd handcheck && pip install -e .
python scripts/reproduce_official.py ./cache ./out PHerc0800
```

This reads the public catalogue, the 6 organizer meshes of PHerc0800 on volume 20250521135224 (8.64 um) and the level-4 masked CT, which is used for the centre track and the frame check. The run took about 5 minutes on a fresh cache. Expected output:

```
PHerc0800 20251028213516-auto_grown_20251028213516 OK MEDIUM False zero 0.000 out 0.000
PHerc0800 20251028220042-auto_grown_20251028220042 OK MEDIUM False zero 0.000 out 0.000
PHerc0800 20251028220955-auto_grown_20251028220955 OK MEDIUM False zero 0.000 out 0.000
PHerc0800 20251028222030-auto_grown_20251028222030 OK MEDIUM False zero 0.000 out 0.000
PHerc0800 20251028225813-auto_grown_20251028225813 OK MEDIUM False zero 0.000 out 0.000
PHerc0800 20251029010146-auto_grown_20251029010146 OK MEDIUM False zero 0.000 out 0.000
6 PHerc0800 OK MEDIUM False
```

The columns are: state, confidence, `mirrored_as_stored`, and the frame-check fractions (zero CT, outside the volume). PHerc0800 is MEDIUM because its normal-line centre is ill-conditioned (condition number > 10), which leaves the CT-mask centroid as the single centre source. `out/official_meshes.csv` holds one row per mesh.

## CLI

```bash
handcheck MESH_OR_DIR [...] --props props.json (--axes zyx | --zattrs VOLUME/.zattrs)
          [--umbilicus umbilicus.json] [--centroid centre.json]
          [--frame frame.json | --ct-zarr VOLUME.zarr --ct-level 4]
          [--fragment [--text-side -y]] [--json out.json] [--csv out.csv] [--strict]
```

**Inputs**
- `MESH_OR_DIR`: tifxyz directories (`x.tif`, `y.tif`, `z.tif`, `meta.json`, optional `mask.tif`, read as the official loader does), or folders searched recursively for them. Grid convention: `points[v, u] = (x, y, z)` in level-0 voxels; u is the column, drawn to the right, and v is the row, drawn downward.
- `--props`: the volume's `properties` from the catalogue (`metadata.json` at the bucket root, `samples/<scroll>/volumes/<id>/properties`). Keys used: `left_handed_coordinates`, `z_direction_is_top_to_bottom`, and `text_side` for fragments.
- `--axes` / `--zattrs`: the array axis names. They must be a permutation of x, y, z; otherwise the parity is unknown.
- Centre sources, in level-0 voxels of the same volume:
  - `--umbilicus`: an official umbilicus JSON (`control_points`).
  - `--centroid`: a list of `[x, y, z]`, or `{"z": [...], "x": [...], "y": [...]}`.
  - The mesh's own normal-line centre is always computed. It is used only when well conditioned (condition number <= 10).
- Frame check:
  - `--frame`: precomputed fractions per mesh name.
  - `--ct-zarr`: a local masked OME-Zarr (needs `zarr`), or `s3://vesuvius-challenge-open-data/<scroll>/volumes/<volume>.zarr`, read over HTTPS.

**Outputs**: one line per mesh on stdout, `--csv` one row per mesh, and `--json` the full detail. CSV columns:

| column | meaning |
|---|---|
| `mesh` | tifxyz directory name |
| `state` | `OK`, `FLAG`, `FRAME_UNVERIFIED` or `PARITY_UNKNOWN` |
| `confidence` | `HIGH`, `MEDIUM` or `FLAG` (definitions below) |
| `mirrored_as_stored` | `True` if the grid rendered as stored (u right, v down) reads mirrored, `False` if not; empty if not decided |
| `mirror_u`, `flip_v` | column and row flips that make the render read unmirrored and upright; `flip_v` is empty when up/down is unknown |
| `rot180_equivalent` | `True` when up/down is unknown: the render is unmirrored either way, but may be upside down |
| `parity`, `chirality` | frame parity (+1/-1, 0 = unknown) and the consensus chirality of the mesh |
| `reversed_frac` | fraction of vertices whose chirality is reversed under every centre source (folds, sheet jumps) |
| `frac_zero_ct`, `frac_outside` | frame check: sampled vertices on zero (masked) CT, and outside the volume |
| `sources_used`, `flags` | centre sources that voted, and every reason for a lowered confidence |

**Exit codes**: `0` success; `1` runtime error, such as an unreadable file or no meshes; `2` usage error; `3` with `--strict`, at least one mesh was not decided.

**Python API**

```python
from handcheck.rule import VolumeInfo, FrameCheck, orient_for_reading
from handcheck.tifxyz import read_tifxyz
vi = VolumeInfo.from_metadata(zattrs, props, umbilicus=curve)   # or centroid=..., sample_type="fragment"
d = orient_for_reading(read_tifxyz(path).points, vi, frame=FrameCheck(0.0, 0.0))
d.state, d.confidence, d.mirror_u, d.flip_v, d.flags
img_readable = d.apply(img)      # apply the decided flips to any (H, W, ...) array on the mesh grid
```

## How it decides

- **F1**: the recto is rolled facing the umbilicus. The organizers state that it is "always rolled facing inward, toward the center of the scroll" ([2026 Open Problems](https://scrollprize.org/2026_open_problems)).
- **F2**: with u to the right and v downward, `dP/du x dP/dv` points into the screen. The viewer is therefore on the umbilicus side if and only if `det[dP/du, dP/dv, r_out] > 0` in a right-handed frame. The official renderer states the same convention: with "text readable, U winding outside->inside" the normal `dP/dU x dP/dV` "points toward the OUTSIDE of the scroll" (`volume-cartographer/apps/src/vc_render_tifxyz.cpp` in [ScrollPrize/villa](https://github.com/ScrollPrize/villa)).
- **F3**: the index frame has parity -1 if the catalogue says `left_handed_coordinates: true`, and +1 if false. The render is legible if and only if chirality equals parity. The acquisition z direction is not a mirror: a scroll mounted upside down is a rotation.
- **F4**: fragments use `-text_side` in place of `r_out`.
- **Up/down**: this is a 180-degree ambiguity, never a mirror. If `z_direction_is_top_to_bottom` is true, image top = low z; if false, high z; if null, it is unknown (`rot180_equivalent`).

**Gates, in order:**
1. Unknown parity gives `PARITY_UNKNOWN`.
2. A failed frame check (more than 5 % of sampled vertices on zero CT, or more than 1 % outside the volume) gives `FRAME_UNVERIFIED`.
3. No valid centre source gives `FLAG`.
4. Valid sources disagreeing on the whole-mesh sign gives `FLAG`.
5. Vertices reversed under every source: up to 20 % are masked out of `keep`; more than 20 % gives `FLAG`.

**Confidence levels:**
- **HIGH**: at least 2 valid sources, each agreeing on at least 0.98 of its clear vertices, pairwise per-vertex agreement at least 0.95, and at most 2 % reversed.
- **MEDIUM**: every source at least 0.9 with pairwise agreement at least 0.8, or a single source at least 0.95 with at most 2 % reversed.
- Anything else is `FLAG`.

## Validation

All numbers below are produced by the scripts and tests in this repository.

**Tests.** `pip install -e ".[test]" && pytest` gives **94 passed, 1 skipped**, including the CLI end to end on tifxyz files written to disk and the optional zarr frame check. The one skip is intended: K is nearly up-down symmetric, so the up/down test runs on R only.

**Planted glyphs** (`python scripts/synthetic_validation.py results/synthetic_validation.json`):
- **Scene.** A synthetic Archimedean spiral scroll with 4.5 turns and a 9-voxel pitch. A font-rendered capital R or K is planted as an ink bump 1.2 voxels inside the recto face of one winding, reading correctly from the umbilicus side.
- **32 cases.** There are 2 glyphs x 4 frames x 4 input parameterisations:
  - the frames are ascending, upside-down (a proper 180-degree rotation), reversed slice order (left-handed) and in-plane reflection (left-handed);
  - the parameterisations are as built, u-mirrored, v-flipped, and both.
- **Method.** Each case is levelled, decided by the rule with the CT-mask centroid as centre, rendered along normals, and correlated (normalised cross-correlation, `cv2.TM_CCOEFF_NORMED`) with the glyph and its mirror image.
- **Rule.** **32/32 read unmirrored.** Minimum NCC with the true glyph is **0.954**, and the minimum margin over the mirror image is **0.262** (pass bar: NCC > 0.8 and margin > 0.2).
- **Wrong-sign control.** The opposite mirror decision on the same 32 cases comes out **mirrored 32/32** (minimum margin 0.262).
- **Other synthetic checks** (tests): fragments, both input handednesses; up/down unknown, where both 180-degree candidates are unmirrored; the frame gate; masking of a reversed patch (sheet jump) at 10 % and a FLAG at 30 %; an ill-conditioned flat sheet; source disagreement; and the winding-sense up/down check in all four frames.

**Official meshes** (`python scripts/reproduce_official.py ./cache ./results`). The rule is run on every organizer tifxyz mesh on the listed volume, with a level-4 frame check on every mesh. The inputs come from public data only; the table in `results/official_meshes.csv` is summarised below.

| scroll (volume, voxel size) | meshes | decided: unmirrored as stored | decided: mirrored | not decided |
|---|---|---|---|---|
| PHerc0139 (20250728140407, 9.362 um) | 38 | **38** (38 HIGH) | 0 | 0 |
| PHerc0800 (20250521135224, 8.64 um) | 6 | **6** (6 MEDIUM) | 0 | 0 |
| PHerc1447 (20250521151220, 8.64 um) | 15 | **4** (4 MEDIUM) | 0 | **11**: 10 `FRAME_UNVERIFIED`, 1 `FLAG` |
| PHerc0009B, fragment (20250521125136, 8.64 um) | 18 | **7** (7 MEDIUM) | 0 | **11** `FRAME_UNVERIFIED` (see Findings) |

**Independent up/down check on PHerc0139** (`python scripts/winding_check.py ./cache results/winding_PHerc0139.json`). Rolls read from the outside inward. So the rotation sense of the inward direction about +z, measured along mesh rows or columns that span more than one turn, together with the frame parity, predicts whether the text top is at low z. This agrees with the catalogue flag `z_direction_is_top_to_bottom` on **36/36** meshes that span more than one turn; 2 of the 38 span less than one turn and are not testable.

**Centre sources used:**
- PHerc0139: the official umbilicus (annotated on volume 20260102150214, mapped by the organizer affine) and the mesh normal-line centre.
- PHerc0800 and PHerc1447: the level-4 CT-mask centroid track (every 8th level-4 slice, z-smoothed with sigma 256 voxels) and the normal-line centre where well conditioned.
- PHerc0009B (fragment): `-text_side` (`-y`).

In every decided case the organizer grid reads **unmirrored as stored**. Parity is +1 on every volume: `left_handed_coordinates` is false and the axes are (z, y, x).

## Findings on real data

1. **PHerc1447: 10 of 15 organizer meshes are not on the scroll in the volume frame they declare.**
   - These are the 10 traces dated 2025-05-02 (segment IDs `20250502180708` ... `20250502185519` in `results/official_meshes.csv`), at z 20,396-24,974.
   - 41-91 % of their sampled vertices lie on zero (masked-out) CT, and 0-19 % lie outside the volume.
   - The other 5 meshes of 1447 lie 0-1.4 % on zero CT, and every PHerc0139 control lies at most 3.6 %.
   - Their chirality votes look internally consistent. Without the frame gate, 3 of them would be rated HIGH and 2 MEDIUM, all "unmirrored" (columns `*_without_frame_gate`). A handedness check that skips the frame test would pass them.
   - **Any render or ink prediction made from these 10 meshes in this volume should be treated as uninformative until their frame is confirmed.**
   - The catalogue dates the 10 traces 2025-05-02. The volume they declare (`original_volume_id` 20250521151220) carries the export date 2025-05-21 in its ID, and the catalogue lists no volume transform for this scroll.
   - A plausible explanation is that they were traced in an earlier reconstruction of the scan. This is a hypothesis, not verified.
   - These are **flags to be confirmed by the organizers, not established errors.**
2. **PHerc1447, mesh `20250502205333`: mixed chirality.**
   - The CT-mask centroid and the mesh centre agree on only 75 % of vertices (centroid self-agreement 0.76), and 3.1 % of the vertices are reversed under both.
   - Parts of this mesh would render locally mirrored under any single global flip. The tool reports FLAG ("read both mirror states"), not a guess.
3. **PHerc0009B (fragment): the frame gate is too strict for fragment surfaces.**
   - 11 of 18 meshes have 5.1-12.1 % of sampled vertices on zero CT; the gate is 5 %. None has more than 0.05 % outside the volume.
   - This is unlike the 41-91 % of the PHerc1447 traces.
   - Most likely a fragment's traced recto lies on the boundary of the organizer object mask. This is a hypothesis, not verified.
   - A level-2 check of mesh `20250510172639` gave 7.2 % (level 4: 6.5 %), so coarse sampling alone does not explain it. To rerun it:

     ```bash
     handcheck DIR --props P --axes zyx --fragment --ct-zarr s3://vesuvius-challenge-open-data/PHerc0009B/volumes/20250521125136-8.640um-1.2m-116keV-masked.zarr --ct-level 2
     ```
   - Without the frame gate, all 18 fragment meshes are decided MEDIUM and **unmirrored as stored**, with `-text_side` agreement 1.000 (columns `*_without_frame_gate`).
   - The gate is kept unchanged in this release, so these meshes stay flagged. A fragment-specific frame test is future work.
4. **Every decided organizer mesh reads unmirrored as stored.** No decided mesh on these four volumes needs a flip, so the organizer pipeline's convention holds wherever it can be checked. Up/down: PHerc0139, PHerc1447 and PHerc0009B have text top at low z; PHerc0800 has text top at high z (`flip_v` = False in all cases, because the grids already run in the flagged direction).

## Limitations and known failure modes

- **No letter-level confirmation on real data.** The rule rests on F1 (organizer documentation), F2 (geometry, and the organizer renderer's stated convention) and the catalogue parity flag. The real-data evidence is:
  - the consistency of independent centre sources, per mesh and per vertex;
  - the frame check;
  - the winding-sense up/down check on PHerc0139 (36/36, `scripts/winding_check.py`).

  No rendered text was inspected to produce these results.
- **Fragments** depend on the meaning of the catalogue `text_side` field, a single source. Fragment decisions are therefore at most MEDIUM.
- **Oval or crushed scrolls.** The CT-mask centroid is only a proxy for the umbilicus. On strongly oval cross-sections it can point the wrong way locally. The per-vertex cross-check with the mesh normal-line centre catches this, and the tool then flags instead of deciding.
- **Flat or nearly flat meshes** have near-parallel normals. Their normal-line centre is ill-conditioned (condition number > 10) and is not used, so such a mesh is decided from a single source (MEDIUM at best).
- **The frame check needs the masked CT.** Without it, the output carries the flag `frame not checked`. A mesh from another volume, or with a wrong affine, would then be decided in the wrong frame.
- **Up/down** is only as good as `z_direction_is_top_to_bottom`. When that is null, the tool reports both 180-degree candidates.
- The thresholds (5 % / 1 % frame gate, 20 % / 2 % reversed, 0.98 / 0.95 / 0.9 / 0.8 agreement) were set on the synthetic tests and the PHerc0139 controls. They are not learned.

## Data and citation

This tool was run on the newer scans released by Vesuvius Challenge: PHerc0139, PHerc0009B, PHerc0800 and PHerc1447, with volume and segment IDs as listed in `results/official_meshes.csv`. Browse them in the [Data Browser](https://scrollprize.org/data_browser); data: https://scrollprize.org/data.

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge.

## Licences

- Code: MIT (`LICENSE`).
- Per-segment result tables in `results/` are derived from Vesuvius Challenge data and licensed CC BY-NC 4.0 (`data/LICENSE-DATA.md`).
- No CT volume, surface volume or mesh is stored here.

## Credits

- The tifxyz format, the official loader semantics mirrored in `handcheck/tifxyz.py`, and the renderer convention quoted above: VC3D / volume-cartographer in [ScrollPrize/villa](https://github.com/ScrollPrize/villa).
- NumPy: C. R. Harris et al. "Array programming with NumPy." Nature 585, 357-362 (2020).
- SciPy (`ndimage`): P. Virtanen et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python." Nature Methods 17, 261-272 (2020).
- tifffile: C. Gohlke, https://github.com/cgohlke/tifffile.
- OpenCV (tests only: glyph rendering and normalised cross-correlation): G. Bradski. "The OpenCV Library." Dr. Dobb's Journal of Software Tools (2000).
- zarr-python (optional): https://github.com/zarr-developers/zarr-python.

## AI assistance

Developed with the help of AI coding agents; all numbers are produced by the scripts and tests in this repository.

## Contributing and issues

Bug reports, questions and meshes where the prediction looks wrong are welcome as [GitHub issues](https://github.com/abundantjoe/handcheck/issues). Please include the segment ID, the volume ID and the `--json` output.
