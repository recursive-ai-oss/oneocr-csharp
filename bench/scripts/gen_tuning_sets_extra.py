"""Usage: gen_tuning_sets_extra.py OUT

A second family of tuning sets for page-stage rules: degraded scans, screenshots, complex layout, clean print and phone photos,
rendered under seeds that no frozen or earlier tuning set used, into OUT/<name>-tune2. Like gen_tuning_sets.py, they are for choosing
thresholds that must not be chosen on a scoring set; the frozen sets, their manifests and their seeds are not touched. Needs the
bench data directory (ONEOCR_BENCH_DIR with `fonts/` and `text/`) and `uv run` from bench/.
"""

import dataclasses
import sys
from pathlib import Path

from oneocr_bench import sets

SEEDS = {
    "degraded-v1": ("degraded-tune2", 20290301, 40),
    "screenshot-v1": ("screenshot-tune2", 20290302, 60),
    "layout-v1": ("layout-tune2", 20290303, 40),
    "clean300-v1": ("clean300-tune2", 20290304, 24),
    "phone-v1": ("phone-tune2", 20290305, 24),
}

out = Path(sys.argv[1])
for source, (name, seed, count) in SEEDS.items():
    sets.SETS[source] = dataclasses.replace(sets.SETS[source], seed=seed)
    manifest = sets.render(source, count, out / name)
    print(f"{name}: {len(manifest['pages'])} pages in {out / name}")
