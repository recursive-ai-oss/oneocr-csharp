"""Usage: orient_trace.py BENCH_DLL MODEL TAG FILE...

Runs `OneOcr.Bench skew --image F --model MODEL` per file and prints, per file, the turn decided and the first axis's
confidences (upright, turned half way round), so the evidence can be compared between upright and rotated pages.
"""
import re
import subprocess
import sys

dll, model, tag = sys.argv[1:4]
for f in sys.argv[4:]:
    out = subprocess.run(["dotnet", dll, "skew", "--image", f, "--model", model], capture_output=True, text=True).stdout
    axes = re.findall(r"axis (\d): upright ([\d.]+) turned ([\d.]+) share ([\d.]+)", out)
    turn = re.search(r"turn (\d+)", out)
    contrast = re.search(r"contrast flat ([\d.]+) sideways ([\d.]+)", out)
    first = axes[0] if axes else ("?", "0", "0", "0")
    print(tag, f.rsplit("/", 1)[-1], "turn", turn.group(1) if turn else "?", "axes", len(axes), "first", first[0], first[1], first[2],
          "delta %.3f" % (float(first[2]) - float(first[1])), "share", first[3], "contrast", contrast.groups() if contrast else "?")
