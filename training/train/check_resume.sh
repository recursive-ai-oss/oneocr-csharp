#!/usr/bin/env bash
# Checks that an interrupted-and-resumed run equals an uninterrupted one: 400 steps straight vs. 400 steps killed (SIGKILL) after
# step 250 and restarted from the last checkpoint. Uses bit-reproducible kernels and the CPU CTC loss (train.deterministic).
#   train/check_resume.sh CONFIG.yaml WORK_DIR        (CONFIG: a training config; its data is only read)
set -uo pipefail
cd "$(dirname "$0")/.."
cfg=$1
T=$2
rm -rf "$T"; mkdir -p "$T"
T=$(cd "$T" && pwd)
export T CFG=$cfg
uv run python - <<'P'
import os, yaml
T = os.environ['T']
c = yaml.safe_load(open(os.environ['CFG']))
c = __import__('json').loads(os.path.expandvars(__import__('json').dumps(c)))
c['train'].update(steps=400, warmup=50, eval_every=200, ckpt_every=100, log_every=50, deterministic=True)
for n in ('a', 'b'):
    c['run_dir'] = f'{T}/{n}'
    yaml.safe_dump(c, open(f'{T}/{n}.yaml', 'w'))
P
uv run python -m train.run "$T/a.yaml" > "$T/a.log" 2>&1
uv run python -m train.run "$T/b.yaml" > "$T/b1.log" 2>&1 &
pid=$!
until grep -q '"step": 250,' "$T/b/metrics.jsonl" 2>/dev/null; do sleep 0.3; done
pkill -9 -f "train.run $T/b.yaml"; wait $pid 2>/dev/null; sleep 3  # uv run is a wrapper: kill the python child too
uv run python -m train.run "$T/b.yaml" > "$T/b2.log" 2>&1
grep -h resumed "$T/b2.log"
uv run python - <<'P'
import json, os, torch
T = os.environ['T']
a = torch.load(f'{T}/a/final.pt', map_location='cpu', weights_only=False)
b = torch.load(f'{T}/b/final.pt', map_location='cpu', weights_only=False)
same = all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
d = max((a['model'][k].double() - b['model'][k].double()).abs().max().item() for k in a['model'])
same_opt = all(torch.equal(a['opt']['state'][k][n], b['opt']['state'][k][n]) for k in a['opt']['state'] for n in ('exp_avg', 'exp_avg_sq'))
print('final weights bit-identical:', same, '| optimizer state bit-identical:', same_opt, '| max abs weight diff', d)
la = {r['step']: r['loss'] for r in map(json.loads, open(f'{T}/a/metrics.jsonl')) if 'loss' in r}
lb = {r['step']: r['loss'] for r in map(json.loads, open(f'{T}/b/metrics.jsonl')) if 'loss' in r}
print('loss per logged step, uninterrupted vs resumed:', {s: (round(la[s], 6), round(lb[s], 6)) for s in sorted(la)})
P
