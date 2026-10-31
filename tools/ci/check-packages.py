#!/usr/bin/env python3
"""Checks the packages produced by tools/ci/pack.sh against what a release must contain.

    python3 tools/ci/check-packages.py <directory with the .nupkg and .snupkg files> <version> [--no-commit-check]

--no-commit-check is for a checkout without git metadata (the repository commit is then missing from the package).
Exit status 1 and a list of problems when a package is wrong. Per package it checks: all seven packages exist at the
version, metadata (id, version, authors, copyright, licence expression, repository and project URL, readme, description),
contents (README.md naming the package, THIRD-PARTY-NOTICES.md, both licence texts, the assembly and its XML documentation),
that the Models packages carry the pinned weights (name, size and SHA-256 from models.json) and that the others carry none,
that the core package depends on RecursiveAI.OneImage and the Models package only, and that a symbols package exists.
"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

PREFIX = "RecursiveAI."
REPO = "https://github.com/recursive-ai-oss/oneocr-csharp"
NAMES = ["OneOcr", "OneOcr.Llm", "OneOcr.Training", "OneOcr.Pdf", "OneOcr.Models", "OneOcr.Models.Documents", "OneOcr.Tool"]
WEIGHTS = {"OneOcr.Models": "src/OneOcr.Models/models.json", "OneOcr.Models.Documents": "src/OneOcr.Models.Documents/models.json"}
# The only package references a package may carry (the declared dependency floors).
ALLOWED_DEPS = {
    "OneOcr": {"RecursiveAI.OneImage", "RecursiveAI.OneOcr.Models"},
    "OneOcr.Llm": {"RecursiveAI.OneOcr"},
    "OneOcr.Training": {"RecursiveAI.OneOcr", "RecursiveAI.OneOcr.Llm"},
    "OneOcr.Pdf": {"RecursiveAI.OneOcr", "RecursiveAI.OnePdf", "RecursiveAI.OneImage.Pdf"},
    "OneOcr.Models": set(),
    "OneOcr.Models.Documents": {"RecursiveAI.OneOcr"},
    "OneOcr.Tool": set(),  # a tool carries its assemblies
}

problems = []


def fail(pkg, msg):
    problems.append(f"{pkg}: {msg}")


def strip_namespaces(root):
    for e in root.iter():
        e.tag = e.tag.rpartition("}")[2]


def check(path: Path, short: str, version: str, repo_root: Path):
    name = path.name
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        nuspec_name = next((n for n in names if n.endswith(".nuspec")), None)
        if nuspec_name is None:
            fail(name, "no nuspec")
            return
        root = ElementTree.fromstring(z.read(nuspec_name))
        strip_namespaces(root)
        meta = root.find("metadata")

        def text(tag):
            e = meta.find(tag)
            return e.text.strip() if e is not None and e.text else ""

        if text("id") != PREFIX + short:
            fail(name, f"id is '{text('id')}'")
        if text("version") != version:
            fail(name, f"version is '{text('version')}', want {version}")
        lic = meta.find("license")
        if lic is None or lic.get("type") != "expression" or (lic.text or "").strip() != "MIT OR Apache-2.0":
            fail(name, f"licence expression is '{lic.text if lic is not None else None}'")
        if len(text("description")) < 20:
            fail(name, "description missing or too short")
        if text("authors") != "Recursive AI, LLC":
            fail(name, "authors")
        if "Recursive AI, LLC" not in text("copyright"):
            fail(name, "copyright")
        repo = meta.find("repository")
        if repo is None or repo.get("type") != "git" or not (repo.get("url") or "").startswith(REPO):
            fail(name, f"repository url is '{repo.get('url') if repo is not None else None}'")
        elif not repo.get("commit") and "--no-commit-check" not in sys.argv:
            fail(name, "repository commit missing")
        if not text("projectUrl").startswith(REPO):
            fail(name, "projectUrl")
        if text("readme") != "README.md":
            fail(name, "readme not declared")
        if not text("tags"):
            fail(name, "no tags")

        deps = {d.get("id") for d in meta.iter("dependency")}
        extra = deps - ALLOWED_DEPS[short]
        if extra:
            fail(name, f"unexpected dependencies {sorted(extra)}")
        missing = ALLOWED_DEPS[short] - deps
        if missing:
            fail(name, f"missing dependencies {sorted(missing)}")

        for need in ["README.md", "THIRD-PARTY-NOTICES.md", "licenses/LICENSE-MIT", "licenses/LICENSE-APACHE"]:
            if need not in names:
                fail(name, f"missing {need}")
        if short != "OneOcr.Tool":
            for need in (f"lib/net10.0/{short}.dll", f"lib/net10.0/{short}.xml"):
                if need not in names:
                    fail(name, f"missing {need}")
        elif not any(n.startswith("tools/net10.0/any/") and n.endswith("OneOcr.Tool.dll") for n in names):
            fail(name, "no tool assembly")
        if "README.md" in names:
            readme = z.read("README.md").decode("utf-8")
            if len(readme) < 200 or (PREFIX + short not in readme and short != "OneOcr.Tool"):
                fail(name, "README.md does not describe this package")
            if short == "OneOcr.Tool" and "oneocr" not in readme:
                fail(name, "README.md does not describe the tool")
        for n in ("README.md", nuspec_name):
            if n in names:
                body = z.read(n).decode("utf-8", "replace")
                if re.search(r"/home/[A-Za-z]|/Users/[A-Za-z]|PLAN\.md|DESIGN\.md|\[(Agent default|Owner|PM)\b", body):
                    fail(name, f"{n} mentions an internal path or document")

        # Weights: the Models packages embed exactly the pinned files inside their assembly; the others embed none.
        dll = next((n for n in names if n.endswith(f"{short}.dll")), None)
        blob = z.read(dll) if dll else b""
        if short in WEIGHTS:
            for m in json.load(open(repo_root / WEIGHTS[short]))["models"]:
                # Each weight is embedded verbatim as a managed resource, so its SHA-256 is found by hashing the stretch
                # that starts at the resource's first bytes.
                found = _find_resource(blob, m["bytes"], m["sha256"])
                if not found:
                    fail(name, f"{m['asset']} ({m['bytes']} bytes, {m['sha256'][:12]}...) is not embedded in {short}.dll")
        elif dll and len(blob) > 5_000_000:
            fail(name, f"{short}.dll is {len(blob)} bytes: weights embedded where none belong")

    symbols = path.with_suffix(".snupkg")
    if not symbols.exists():
        fail(name, "no symbols package (.snupkg)")
    else:
        with zipfile.ZipFile(symbols) as z:
            if not any(n.endswith(".pdb") for n in z.namelist()):
                fail(name, "symbols package holds no .pdb")


def _find_resource(blob: bytes, size: int, sha: str) -> bool:
    """True when some stretch of the assembly with the given length hashes to sha. Resources are stored with a 4-byte length
    prefix, so the candidate stretches start right after a little-endian copy of the size."""
    prefix = size.to_bytes(4, "little")
    start = 0
    while True:
        i = blob.find(prefix, start)
        if i < 0:
            return False
        stretch = blob[i + 4:i + 4 + size]
        if len(stretch) == size and hashlib.sha256(stretch).hexdigest() == sha:
            return True
        start = i + 1


def main():
    d = Path(sys.argv[1])
    version = sys.argv[2]
    repo_root = Path(__file__).resolve().parents[2]
    for short in NAMES:
        p = d / f"{PREFIX}{short}.{version}.nupkg"
        if not p.exists():
            fail(p.name, "not found")
            continue
        check(p, short, version, repo_root)
        print(f"{p.name}  {p.stat().st_size:>10,} bytes")
    others = {p.name for p in d.glob("*.nupkg")} - {f"{PREFIX}{s}.{version}.nupkg" for s in NAMES}
    if others:
        fail("directory", f"unexpected packages {sorted(others)}")
    if problems:
        print()
        print("\n".join(problems))
        print(f"{len(problems)} problem(s).")
        return 1
    print("7 packages checked, no problems.")
    return 0


sys.exit(main())
