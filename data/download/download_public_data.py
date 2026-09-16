"""Stream public accession files with resumable output and optional SHA-256 checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "iot-reproducibility/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = existing > 0 and getattr(response, "status", 200) == 206
        if not append:
            existing = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[2] / "configs/datasets/download_manifest.json")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(args.dataset or [])
    records = []
    for item in manifest["files"]:
        if selected and item["dataset"] not in selected:
            continue
        destination = args.root / item["destination"]
        if args.dry_run:
            records.append({"dataset": item["dataset"], "status": "planned", "url": item["url"], "destination": str(destination)})
            continue
        if not destination.exists():
            _download(item["url"], destination)
        digest = _sha256(destination)
        expected = item.get("sha256")
        if expected and digest != expected:
            raise RuntimeError(f"checksum mismatch for {item['dataset']}: {digest} != {expected}")
        records.append({"dataset": item["dataset"], "status": "verified_local", "url": item["url"],
                        "destination": str(destination), "sha256": digest, "size_bytes": destination.stat().st_size})
    output = args.root / "data" / "download" / "download_manifest_runtime.json"
    output.write_text(json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
