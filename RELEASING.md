# Releasing

A release is one tag. Everything after the tag runs in GitHub Actions; nobody pushes packages by hand and no API key is stored anywhere.

## How the pieces fit

- **Version.** `<Version>` in `Directory.Build.props` is the one place. The tag must be `v` plus that version; the release workflow stops otherwise.
- **Weights.** The model files are never committed. `src/OneOcr.Models/models.json` and `src/OneOcr.Models.Documents/models.json` pin each file by name, size and SHA-256. The files themselves are assets of the GitHub release named in `tools/ci/weights-release.txt`: the release that first published them (a draft until the packages were on nuget.org, which keeps unreleased weights invisible to everyone without write access). CI and the release workflow download the assets from that release, check every file against its pin, and the build checks the hash again before embedding a file. A release whose weights did not change uploads nothing: 1.1.0 uses the same five files as 1.0.0 (`crnn-v0-0.7.3.onm`, `dbnet-lite-v1-0.3.2.onm`, `charlm-v1.bin`, `wordfreq-v2.bin`, `docextract-v0-0.1.0.onm`), and `weights-release.txt` says `v1.0.0`. New weights mean new assets on a new draft release, new pins, and `weights-release.txt` pointing at that tag.
- **Package validation.** Every pack compares each package's public API with the last release (`PackageValidationBaselineVersion` in `src/Directory.Build.props`, 1.0.0 for 1.1.x) and fails on a breaking change, so a minor release is additive by construction. Move the baseline to the previous release when you cut the next minor version.
- **CI** (`.github/workflows/ci.yml`) runs on pushes to `main` and `ci-check/**` on win-x64, linux-x64, linux-arm64 and osx-arm64: Release build with warnings as errors, the test suite, and the NativeAOT smoke app. On Linux it also packs the seven packages, checks them (`tools/ci/check-packages.py`) and runs them from a fresh program on the JIT and as a NativeAOT binary (`tools/ci/consumer-smoke.sh`). The weights come from the draft release for the current version. On a `ci-check/**` branch a missing release fails the run; on `main` the run continues without weights and the tests that need them skip.
- **Release** (`.github/workflows/release.yml`) runs on a `v*` tag: it checks the tag, that the commit is on `main`, that CI passed on that exact commit, and that `CHANGELOG.md` has the version; fetches and verifies the weights; builds, tests, packs and checks; consumes the packages from a fresh program; attests provenance; pushes to nuget.org with Trusted Publishing (`NuGet/login`, no secret); attaches the packages to the draft release and publishes it with the changelog section as notes. The GitHub release goes public only after the packages are on nuget.org.

## One-time setup

1. Repository: Actions enabled. Environment `nuget` (the publish job uses it; add required reviewers if you want a manual gate there too).
2. nuget.org: a Trusted Publishing policy for the repository `recursive-ai-oss/oneocr-csharp`, workflow `release.yml`, environment `nuget`, covering `RecursiveAI.OneOcr*`. The `user` in `release.yml` is the nuget.org account that owns the policy.
3. The repository is public before the tag: build provenance attestations need it.

## Releasing a version

Run from a clean checkout of the commit to release (replace `1.1.0`). `W` is a directory holding the weight files under the asset names in the manifests (download them from the release in `tools/ci/weights-release.txt`, or run `GH_TOKEN=... tools/ci/fetch-weights.sh v1.0.0 "$W"`, which also checks every file against its pin); the pack in step 2 refuses a file whose SHA-256 differs from its pin.

```sh
# 1. Version and notes are in the commit: Directory.Build.props <Version> and a "## [1.1.0] - date" section in CHANGELOG.md
#    (replace "unreleased" by the release date; the section's text becomes the release notes).

# 2. Check the weights and the packages locally if you like (needs the dotnet SDK from global.json and network access for the validation baseline).
tools/ci/pack.sh /tmp/packages "$W" && python3 tools/ci/check-packages.py /tmp/packages 1.1.0
tools/ci/consumer-smoke.sh /tmp/packages 1.1.0 linux-x64

# 3. Create the draft release that will carry the packages. It holds no weights when they did not change (nothing is public yet).
gh release create v1.1.0 --repo recursive-ai-oss/oneocr-csharp --draft --title v1.1.0 --notes "draft"

# 4. CI on the exact commit.
git push origin HEAD:refs/heads/ci-check/release-1.1.0        # wait for a green ci run

# 5. Commit and tag go to GitHub in one push.
git tag v1.1.0
git push --atomic origin HEAD:refs/heads/main v1.1.0
```

Release candidates (`1.1.0-rc.N`) are packed with `-p:Version=1.1.0-rc.N` into a local feed and never pushed; the tag is only for the final version.

The consumer smoke test (step 2, and the `pack-and-consume` job in CI) installs the packages into a fresh program, reads an image, makes a searchable PDF from it and checks the PDF with OnePDF's text extraction, reads a standard-mode fax TIFF (`testdata/fax`), and does all of it again as a NativeAOT binary.

The tag triggers `release.yml`. When it finishes, the packages are listed on nuget.org (check
`https://api.nuget.org/v3-flatcontainer/<id>/index.json` for all seven ids, lower case) and the draft release is published with the
packages attached. If a step fails before the push to nuget.org, fix it and push a new commit and tag; the draft stays a draft.
If the push succeeded and a later step failed, re-run the failed job: package pushes skip versions that exist.

A weights-only patch release uses the same flow with new assets on its own draft release (created with the files, as the 1.0.0 release was), new pins in the `models.json` files, `weights-release.txt` pointing at that tag, and a new version.
