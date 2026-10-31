#!/usr/bin/env bash
# Downloads the weight files pinned in src/OneOcr.Models/models.json and src/OneOcr.Models.Documents/models.json from the
# GitHub release of a tag and checks every file's size and SHA-256 against its pin. A draft release works: it is found by
# listing the repository's releases, which a token with write access can do. A file that fails a check is not kept.
#
#   GH_TOKEN=... tools/ci/fetch-weights.sh v1.0.0 <directory>
#
# Exit status: 0 all files fetched and verified; 1 a file is missing or does not match its pin; 3 no release has that tag.
set -euo pipefail
tag=${1:?usage: fetch-weights.sh <tag> <directory>}
out=${2:?usage: fetch-weights.sh <tag> <directory>}
repo=${GITHUB_REPOSITORY:-recursive-ai-oss/oneocr-csharp}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "$out"

release=$(gh api --paginate "repos/$repo/releases?per_page=100" \
  --jq ".[] | select(.tag_name == \"$tag\") | {id, draft, assets: [.assets[] | {id, name, size}]}" | head -n 1)
if [[ -z $release ]]; then
  echo "no release with tag $tag in $repo" >&2
  exit 3
fi

RELEASE="$release" OUT="$out" REPO="$repo" python3 - <<'PY'
import hashlib, json, os, subprocess, sys

release = json.loads(os.environ["RELEASE"])
out, repo = os.environ["OUT"], os.environ["REPO"]
print("release is a draft" if release["draft"] else "release is published", f"with {len(release['assets'])} assets")
assets = {a["name"]: a for a in release["assets"]}
bad = 0
for manifest in ("src/OneOcr.Models/models.json", "src/OneOcr.Models.Documents/models.json"):
    for m in json.load(open(manifest))["models"]:
        name, want_size, want_sha = m["asset"], m["bytes"], m["sha256"]
        a = assets.get(name)
        if a is None:
            print(f"MISSING {name}: not an asset of the release", file=sys.stderr)
            bad += 1
            continue
        tmp = os.path.join(out, name + ".part")
        with open(tmp, "wb") as f:
            subprocess.run(["gh", "api", "-H", "Accept: application/octet-stream",
                            f"repos/{repo}/releases/assets/{a['id']}"], stdout=f, check=True)
        with open(tmp, "rb") as f:
            data = f.read()
        sha = hashlib.sha256(data).hexdigest()
        if len(data) != want_size or sha != want_sha:
            print(f"MISMATCH {name}: {len(data)} bytes, sha256 {sha}; pinned {want_size} bytes, {want_sha}", file=sys.stderr)
            os.remove(tmp)
            bad += 1
            continue
        os.replace(tmp, os.path.join(out, name))
        print(f"ok {name} {len(data)} {sha}")
sys.exit(1 if bad else 0)
PY
