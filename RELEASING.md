# Releasing

A release is one tag. Everything after the tag runs in GitHub Actions; nobody pushes packages by hand and no API key is stored anywhere.

## How the pieces fit

- **Version.** `<Version>` in `Directory.Build.props` is the one place. The tag must be `v` plus that version; the release workflow stops otherwise.
- **Weights.** The model files are never committed. `src/OneOcr.Models/models.json` and `src/OneOcr.Models.Documents/models.json` pin each file by name, size and SHA-256. The files themselves are assets of a **draft GitHub release named after the tag** (`v1.0.0`). A draft is invisible to everyone without write access, so the weights are not published by uploading them, and publishing the draft is the last step of the release. CI and the release workflow download the assets, check every file against its pin, and the build checks the hash again before embedding a file.
- **CI** (`.github/workflows/ci.yml`) runs on pushes to `main` and `ci-check/**` on win-x64, linux-x64, linux-arm64 and osx-arm64: Release build with warnings as errors, the test suite, and the NativeAOT smoke app. On Linux it also packs the seven packages, checks them (`tools/ci/check-packages.py`) and runs them from a fresh program on the JIT and as a NativeAOT binary (`tools/ci/consumer-smoke.sh`). The weights come from the draft release for the current version. On a `ci-check/**` branch a missing release fails the run; on `main` the run continues without weights and the tests that need them skip.
- **Release** (`.github/workflows/release.yml`) runs on a `v*` tag: it checks the tag, that the commit is on `main`, that CI passed on that exact commit, and that `CHANGELOG.md` has the version; fetches and verifies the weights; builds, tests, packs and checks; consumes the packages from a fresh program; attests provenance; pushes to nuget.org with Trusted Publishing (`NuGet/login`, no secret); attaches the packages to the draft release and publishes it with the changelog section as notes. The GitHub release goes public only after the packages are on nuget.org.

## One-time setup

1. Repository: Actions enabled. Environment `nuget` (the publish job uses it; add required reviewers if you want a manual gate there too).
2. nuget.org: a Trusted Publishing policy for the repository `recursive-ai-oss/oneocr-csharp`, workflow `release.yml`, environment `nuget`, covering `RecursiveAI.OneOcr*`. The `user` in `release.yml` is the nuget.org account that owns the policy.
3. The repository is public before the tag: build provenance attestations need it.

## Releasing a version

Run from a clean checkout of the commit to release (replace `1.0.0`). `W` is a directory holding the weight files under the asset names in the manifests; the pack in step 3 refuses a file whose SHA-256 differs from its pin.

```sh
# 1. Version and notes are in the commit: Directory.Build.props <Version> and a "## [1.0.0] - date" section in CHANGELOG.md.

# 2. Create the draft release with the weights (nothing is public yet).
gh release create v1.0.0 --repo recursive-ai-oss/oneocr-csharp --draft --title v1.0.0 --notes "draft" \
  "$W"/crnn-v0-0.7.3.onm "$W"/dbnet-lite-v1-0.3.2.onm "$W"/charlm-v1.bin "$W"/wordfreq-v2.bin "$W"/docextract-v0-0.1.0.onm

# 3. Check the weights and the packages locally if you like (needs the dotnet SDK from global.json).
tools/ci/pack.sh /tmp/packages "$W" && python3 tools/ci/check-packages.py /tmp/packages 1.0.0
tools/ci/consumer-smoke.sh /tmp/packages 1.0.0 linux-x64

# 4. CI on the exact commit.
git push origin HEAD:refs/heads/ci-check/release-1.0.0        # wait for a green ci run

# 5. Commit and tag go to GitHub in one push.
git tag v1.0.0
git push --atomic origin HEAD:refs/heads/main v1.0.0
```

The tag triggers `release.yml`. When it finishes, the packages are listed on nuget.org (check
`https://api.nuget.org/v3-flatcontainer/<id>/index.json` for all seven ids, lower case) and the draft release is published with the
weights and packages attached. If a step fails before the push to nuget.org, fix it and push a new commit and tag; the draft stays a draft.
If the push succeeded and a later step failed, re-run the failed job: package pushes skip versions that exist.

A weights-only patch release uses the same flow with new assets, new pins in the `models.json` files and a new version.
