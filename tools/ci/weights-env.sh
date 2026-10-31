#!/usr/bin/env bash
# Prints the MSBuild properties that embed the weights in <weights directory>, one NAME=path per line, with the file names the
# models.json manifests give. The build reads each name from the environment or from -p:, hashes the file against its pin and
# stops on a mismatch. Plain shell tools only, so it runs on every CI runner.
#
#   tools/ci/weights-env.sh <weights directory> >> "$GITHUB_ENV"
set -euo pipefail
dir=${1:?usage: weights-env.sh <weights directory>}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
prop() {
  case $1 in
    recognizer) echo OneOcrModelFile ;;
    detector) echo OneOcrDetectorFile ;;
    language-model) echo OneOcrLanguageModelFile ;;
    word-frequencies) echo OneOcrWordFrequencyFile ;;
    extractor) echo OneOcrDocumentModelFile ;;
    *) echo "unknown role $1" >&2; return 1 ;;
  esac
}
for manifest in src/OneOcr.Models/models.json src/OneOcr.Models.Documents/models.json; do
  grep -E '"(role|asset)"' "$manifest" | sed -E 's/.*"(role|asset)": *"([^"]*)".*/\2/' | paste -d' ' - - | while read -r role asset; do
    echo "$(prop "$role")=$dir/$asset"
  done
done
