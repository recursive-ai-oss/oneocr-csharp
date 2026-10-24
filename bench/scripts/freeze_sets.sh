#!/usr/bin/env bash
# Freeze generated sets as a read-only evaluation copy with a checksum file.
#
#   ONEOCR_BENCH_DIR=<generated data> bash scripts/freeze_sets.sh DEST
#
# Copies pages/ and truth/ of every set from $ONEOCR_BENCH_DIR to DEST/<set>/, writes DEST/SHA256SUMS (relative paths,
# sorted) and removes write permission. Refuses to overwrite an existing DEST. Verify later with
# `cd DEST && sha256sum -c SHA256SUMS`.
set -euo pipefail
dest=${1:?usage: freeze_sets.sh DEST}
src=${ONEOCR_BENCH_DIR:?set ONEOCR_BENCH_DIR to the generated data directory}
sets="clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1"
[ -e "$dest" ] && { echo "$dest exists; not overwriting" >&2; exit 1; }
mkdir -p "$dest"
for s in $sets; do
  mkdir -p "$dest/$s"
  cp -r "$src/$s/pages" "$src/$s/truth" "$dest/$s/"
done
(cd "$dest" && LC_ALL=C find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
chmod -R a-w "$dest"
echo "froze $(wc -l < "$dest/SHA256SUMS") files into $dest"
