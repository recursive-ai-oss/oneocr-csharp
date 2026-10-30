"""Usage: gen_tight_sets.py OUT [COUNT]

Renders clean-print, layout and degraded-scan pages like clean300-v1, layout-v1 and degraded-v1 but set solid (line height 1.0 to
1.1 of the type size, as statute and report text is), under new seeds, into OUT/clean300-tight, OUT/layout-tight and
OUT/degraded-tight. Lines set this way touch row by row (descenders reach the next line's ascenders), which is what the page stage's
line splitting has to cope with; the frozen sets and the other tuning sets are not touched. Same environment as gen_tuning_sets.py
(ONEOCR_BENCH_DIR with `fonts/` and `text/`).
"""

import dataclasses
import sys
from pathlib import Path

from oneocr_bench import gen, layouts, sets

SEEDS = {"clean300-v1": ("clean300-tight", 20270301), "layout-v1": ("layout-tight", 20270302), "degraded-v1": ("degraded-tight", 20270303)}

gen.LINE_HEIGHTS = [1.0, 1.05, 1.1]
layouts.LINE_HEIGHTS = [1.0, 1.05, 1.1]
out = Path(sys.argv[1])
count = int(sys.argv[2]) if len(sys.argv) > 2 else 24
for source, (name, seed) in SEEDS.items():
    sets.SETS[source] = dataclasses.replace(sets.SETS[source], seed=seed)
    dest = out / name
    manifest = sets.render(source, count, dest)
    print(f"{name}: {len(manifest['pages'])} pages in {dest}")
