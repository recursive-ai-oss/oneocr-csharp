#!/usr/bin/env python3
"""Checks that every code sample in the given Markdown files is the text of a region in Samples.cs.

    python3 tests/OneOcr.Samples/check_readme.py README.md docs/migrating-from-tesseract.md

A sample in Markdown is a fenced `csharp` block directly after a line `<!-- sample: Name -->`. Samples.cs holds the same text
between `#region Name` and `#endregion` (indentation removed), and the project is built, so the README cannot show code that
does not compile against the current API.
"""

import re
import sys
import textwrap
from pathlib import Path

here = Path(__file__).resolve().parent
source = (here / "Samples.cs").read_text(encoding="utf-8")
regions = {}
for m in re.finditer(r"^[ \t]*#region (\w+)\n(.*?)^[ \t]*#endregion", source, re.S | re.M):
    regions[m.group(1)] = textwrap.dedent(m.group(2)).strip("\n")

bad = 0
seen = set()
for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8")
    for m in re.finditer(r"<!-- sample: (\w+) -->\n```csharp\n(.*?)```", text, re.S):
        key, body = m.group(1), m.group(2).strip("\n")
        seen.add(key)
        if key not in regions:
            print(f"{name}: sample {key} has no region in Samples.cs")
            bad += 1
        elif regions[key] != body:
            print(f"{name}: sample {key} differs from its region in Samples.cs")
            bad += 1
    fences = len(re.findall(r"^```csharp", text, re.M))
    marked = len(re.findall(r"<!-- sample: \w+ -->\n```csharp", text))
    if fences != marked:
        print(f"{name}: {fences - marked} csharp block(s) without a sample marker")
        bad += 1

unused = set(regions) - seen
if unused:
    print("regions not used by any file: " + ", ".join(sorted(unused)))
    bad += 1
print("ok" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
