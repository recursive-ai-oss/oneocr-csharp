"""Shared helpers for the dataset fetchers (standard library only).

Every fetcher follows the same shape: download pinned archives (resumable,
size- and SHA-256-checked), extract images while hashing them, write one
index per split, de-duplicate real images across splits by content hash, and
finish with a DONE.json that makes the next run a no-op.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER_AGENT = "oneocr-dataset-fetch/1"

# When an image appears in several splits, the copy in the split that comes
# first here is kept and the copies in later splits are listed for exclusion,
# so evaluation images can never leak into training.
SPLIT_PRIORITY = ("test", "dev", "val", "train")


def data_root(override: str | None = None) -> Path:
    return Path(override or os.environ.get("ONEOCR_DATASETS") or Path.home() / "oneocr" / "datasets")


def load_manifest(name: str) -> dict:
    return json.loads((HERE / "manifests" / f"{name}.json").read_text(encoding="utf-8"))


def sha256_file(path: Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(url: str, dest: Path, size: int | None, sha256: str | None, retries: int = 12) -> str:
    """Fetch `url` to `dest`, resuming a partial file; return the verified SHA-256.

    `size` comes from the server's Content-Length when the manifest was
    written. `sha256` is the pin; when it is empty the observed value is
    returned so it can be pinned (the archives' publishers give no checksums).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_name(dest.name + ".part")
        attempt = 0
        while True:
            have = part.stat().st_size if part.exists() else 0
            if size is not None and have == size:
                break
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:
                        have = 0  # server ignored the range: start over
                    mode = "ab" if have else "wb"
                    got = have
                    last = time.time()
                    with open(part, mode) as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                            got += len(chunk)
                            if time.time() - last > 60:
                                log(f"  {dest.name}: {got / 1e9:.2f} GB" + (f" / {size / 1e9:.2f}" if size else ""))
                                last = time.time()
                if size is None:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                attempt += 1
                if attempt > retries:
                    raise
                log(f"  {dest.name}: {e!r}; retry {attempt}/{retries}")
                time.sleep(min(60, 2**attempt))
        part.rename(dest)
    actual = dest.stat().st_size
    if size is not None and actual != size:
        raise SystemExit(f"{dest.name}: size {actual} != expected {size}")
    digest = sha256_file(dest)
    if sha256 and digest != sha256:
        raise SystemExit(f"{dest.name}: sha256 {digest} != pinned {sha256}")
    return digest


def source_key(manifest: dict) -> str:
    """Identity of what is fetched (urls, sizes, subset choice), ignoring the pins."""
    files = sorted((f["url"], f.get("size")) for f in manifest["files"])
    blob = json.dumps([files, manifest.get("subset")], sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def check_done(ds: Path, manifest: dict) -> dict | None:
    """Return the DONE summary when this dataset directory is complete and still matches the pins."""
    p = ds / "DONE.json"
    if not p.exists():
        return None
    done = json.loads(p.read_text())
    if done.get("source_key") != source_key(manifest):
        return None
    for f in manifest["files"]:
        pin, seen = f.get("sha256"), done["archives"].get(f["id"])
        if pin and pin != seen:
            raise SystemExit(f"{f['id']}: fetched with sha256 {seen}, manifest pins {pin}; remove DONE.json to refetch")
    return done


def split_digest(records: list[dict]) -> str:
    """Digest of (id, image sha256) pairs: pins a split exactly without listing it."""
    lines = sorted(f"{r['id']}\t{r['sha256']}" for r in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def dedup(index: dict[str, list[dict]]) -> dict:
    """Find identical image bytes within and across splits.

    Returns counts plus `drop`: ids to exclude from the lower-priority split
    wherever the same bytes also sit in a higher-priority one.
    """
    rank = {s: i for i, s in enumerate(SPLIT_PRIORITY)}
    by_hash: dict[str, list[tuple[str, str]]] = {}
    for split, recs in index.items():
        for r in recs:
            by_hash.setdefault(r["sha256"], []).append((split, r["id"]))
    within = cross = 0
    drop: dict[str, list[str]] = {s: [] for s in index}
    examples = []
    for h, locs in by_hash.items():
        if len(locs) == 1:
            continue
        splits = {s for s, _ in locs}
        if len(splits) == 1:
            within += len(locs) - 1
            continue
        cross += 1
        best = min(splits, key=lambda s: rank.get(s, 99))
        for s, i in locs:
            if s != best:
                drop[s].append(i)
        if len(examples) < 20:
            examples.append({"sha256": h, "at": [f"{s}/{i}" for s, i in locs][:3], "at_count": len(locs)})
    return {
        "images": sum(len(v) for v in index.values()),
        "unique_hashes": len(by_hash),
        "duplicates_within_split": within,
        "hashes_across_splits": cross,
        "drop": {s: sorted(v) for s, v in drop.items() if v},
        "examples": examples,
    }


def write_index(ds: Path, split: str, recs: list[dict]) -> None:
    d = ds / "index"
    d.mkdir(exist_ok=True)
    with open(d / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(recs, key=lambda r: r["id"]):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def finish(ds: Path, manifest: dict, archives: dict[str, str], index: dict[str, list[dict]], extra: dict) -> dict:
    """Dedup, write dedup.json / splits.json / DONE.json and return the summary."""
    for split, recs in index.items():
        write_index(ds, split, recs)
    dd = dedup(index)
    (ds / "dedup.json").write_text(json.dumps(dd, indent=1))
    total = sum(r["bytes"] for recs in index.values() for r in recs)
    splits = {
        s: {"images": len(recs), "image_bytes": sum(r["bytes"] for r in recs), "digest": split_digest(recs)}
        for s, recs in sorted(index.items())
    }
    summary = {
        "name": manifest["name"],
        "version": manifest["version"],
        "source_key": source_key(manifest),
        "archives": archives,
        "splits": splits,
        "image_bytes_total": total,
        "dedup": {k: v for k, v in dd.items() if k not in ("drop", "examples")}
        | {"dropped_from": {s: len(v) for s, v in dd["drop"].items()}, "examples": dd["examples"][:5]},
        **extra,
    }
    # splits.json is the committable record: counts and digests, no data.
    (ds / "splits.json").write_text(json.dumps(summary, indent=1))
    (ds / "DONE.json").write_text(json.dumps(summary, indent=1))
    return summary


def report(summary: dict) -> None:
    log(f"{summary['name']} {summary['version']}: {summary['image_bytes_total'] / 1e9:.2f} GB of images")
    for s, v in summary["splits"].items():
        log(f"  {s}: {v['images']} images, {v['image_bytes'] / 1e9:.2f} GB, digest {v['digest'][:16]}")
    d = summary["dedup"]
    log(
        f"  dedup: {d['images']} images, {d['unique_hashes']} unique, {d['duplicates_within_split']} within-split dups, "
        f"{d['hashes_across_splits']} hashes across splits, dropped_from={d['dropped_from']}"
    )


def common_args(p) -> None:
    p.add_argument("--root", help="dataset root (default $ONEOCR_DATASETS or ~/oneocr/datasets)")
    p.add_argument("--keep-raw", action="store_true", help="keep downloaded archives after extraction")
    p.add_argument("--force", action="store_true", help="ignore DONE.json and redo extraction")


if __name__ == "__main__":
    sys.exit("helper module; run fetch_cord.py, fetch_doclaynet.py or fetch_publaynet.py")
