"""Usage: gen_tuning_sets.py OUT [COUNT]

Renders a degraded-scan set and a screenshot set like degraded-v1 and screenshot-v1 but under new seeds, into OUT/degraded-tune and
OUT/screenshot-tune. They are for choosing thresholds that must not be chosen on the frozen sets; the frozen sets, their manifests and
their seeds are not touched. Needs the fonts and texts of the bench data directory: set ONEOCR_BENCH_DIR to a directory that has
`fonts/` and `text/` (a symlink to the main data directory's is enough) before running, from bench/ with `uv run`.
"""

import dataclasses
import sys
from pathlib import Path

from oneocr_bench import sets

SEEDS = {"degraded-v1": ("degraded-tune", 20270201), "screenshot-v1": ("screenshot-tune", 20270202)}

out = Path(sys.argv[1])
count = int(sys.argv[2]) if len(sys.argv) > 2 else 40
for source, (name, seed) in SEEDS.items():
    sets.SETS[source] = dataclasses.replace(sets.SETS[source], seed=seed)
    dest = out / name
    manifest = sets.render(source, count, dest)
    print(f"{name}: {len(manifest['pages'])} pages in {dest}")
