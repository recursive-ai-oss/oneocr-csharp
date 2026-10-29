"""Usage: gen_confirm_sets.py OUT

Renders confirm sets for a rule that was fitted on the tuning sets: degraded (80 pages), phone (60), screenshot (40), clean print (24)
and complex layout (24), under seeds that neither the frozen sets nor gen_tuning_sets*.py use, into OUT/<name>-conf. Same environment as
gen_tuning_sets.py (ONEOCR_BENCH_DIR with `fonts/` and `text/`, run from bench/ with `uv run`).
"""

import dataclasses
import sys
from pathlib import Path

from oneocr_bench import sets

PLAN = {
    "degraded-v1": ("degraded-conf", 20270301, 80),
    "phone-v1": ("phone-conf", 20270302, 60),
    "screenshot-v1": ("screenshot-conf", 20270303, 40),
    "clean300-v1": ("clean300-conf", 20270304, 24),
    "layout-v1": ("layout-conf", 20270305, 24),
}

out = Path(sys.argv[1])
for source, (name, seed, count) in PLAN.items():
    sets.SETS[source] = dataclasses.replace(sets.SETS[source], seed=seed)
    manifest = sets.render(source, count, out / name)
    print(f"{name}: {len(manifest['pages'])} pages in {out / name}")
