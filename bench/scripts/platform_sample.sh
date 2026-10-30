#!/usr/bin/env bash
# Cuts the fixed page sample for the cross-platform identity check: two pages from each of ten dev sets
# (the first, and the one a third of the way through the sorted list) plus five photos from the receipt-photo set.
#
#   bash bench/scripts/platform_sample.sh DEV_ROOT RECEIPT_PAGES_DIR OUT_DIR
#
# Pages are copied under the name <set>__<page>.<ext>, so `OneOcr.Bench ocrhash --pages OUT_DIR` hashes them in a fixed order.
set -euo pipefail
dev="${1:?usage: platform_sample.sh DEV_ROOT RECEIPT_PAGES_DIR OUT_DIR}"; rec="${2:?}"; out="${3:?}"
mkdir -p "$out"
pick() { # set dir, count of pages wanted at positions given as fractions
  local set="$1" dir="$2"; shift 2
  mapfile -t files < <(ls "$dir" | grep -Ei '\.(png|jpe?g)$' | LC_ALL=C sort)
  local n=${#files[@]}
  for frac in "$@"; do
    local i=$(( n * frac / 100 )); local f="${files[$i]}"
    cp "$dir/$f" "$out/${set}__$f"
  done
}
for s in clean300-v4 clean-real degraded-real degraded-v4 layout-real layout-v4 phone-v4 receipts-v4 screenshot-real screenshot-v4; do
  pick "$s" "$dev/$s/pages" 0 33
done
pick receipts-gen-dev "$rec" 0 20 40 60 80
pick receipts-gen-dev "$rec" 99
ls "$out" | wc -l
