# Data licence

The files in `results/` that list per-segment numbers (`official_meshes.csv`, `official_meshes.json`) are
**derived from Vesuvius Challenge data, CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/).
They were computed from public scans and meshes of PHerc0139, PHerc0009B, PHerc0800 and PHerc1447
(https://scrollprize.org/data). Cite the data as:

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge - CT Scans of Herculaneum Papyri*. Vesuvius Challenge.

`results/synthetic_validation.json` is produced from synthetic volumes only and is covered by the MIT licence of the code.

No CT volume, surface volume or mesh is stored in this repository; the scripts read them from the public bucket.
