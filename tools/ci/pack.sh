#!/usr/bin/env bash
# Packs the seven packages into <output directory> with the weights from <weights directory> (file names as in the models.json
# manifests, which is how fetch-weights.sh writes them). The build hashes every weight file against its pin and stops on a mismatch.
#
#   tools/ci/pack.sh <output directory> <weights directory> [extra dotnet pack arguments]
#
# Pass --no-build when the projects were just built with the same weights (they are embedded at build time).
set -euo pipefail
out=${1:?usage: pack.sh <output directory> <weights directory>}
w=$(cd "${2:?usage: pack.sh <output directory> <weights directory>}" && pwd)
shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
props=()
while read -r line; do props+=("-p:$line"); done < <(tools/ci/weights-env.sh "$w")
mkdir -p "$out"
for p in OneOcr.Models OneOcr OneOcr.Models.Documents OneOcr.Llm OneOcr.Training OneOcr.Pdf OneOcr.Tool; do
  dotnet pack "src/$p" -c Release "${props[@]}" -o "$out" "$@"
done
