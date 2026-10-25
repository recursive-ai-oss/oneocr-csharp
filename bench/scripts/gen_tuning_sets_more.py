"""Usage: gen_tuning_sets_more.py OUT [COUNT]

Like gen_tuning_sets.py for the categories a degraded-page rule must not hurt: clean print, complex layout and phone photos, rendered
under new seeds into OUT/<name>-tune. For choosing rules and thresholds that must not be chosen on the frozen sets; the frozen sets,
their manifests and their seeds are not touched. Same environment as gen_tuning_sets.py (ONEOCR_BENCH_DIR with `fonts/` and `text/`).
"""

import dataclasses
import sys
from pathlib import Path

from oneocr_bench import sets

SEEDS = {"clean300-v1": ("clean300-tune", 20270203), "layout-v1": ("layout-tune", 20270204), "phone-v1": ("phone-tune", 20270205)}

out = Path(sys.argv[1])
count = int(sys.argv[2]) if len(sys.argv) > 2 else 24
for source, (name, seed) in SEEDS.items():
    sets.SETS[source] = dataclasses.replace(sets.SETS[source], seed=seed)
    dest = out / name
    manifest = sets.render(source, count, dest)
    print(f"{name}: {len(manifest['pages'])} pages in {dest}")
