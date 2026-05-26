"""Build a versioned, deterministic sample pack zip for distribution via GCP.

Produces:

  dist/sas-{drum,instrument}-pack-v{N}.zip

The zip contains, at its root:

  _pack-version.json   <- install marker (read by sas-assistant's PackDownloadService)
  <payload tree>       <- drums: drum-role folders; instruments: instruments/<cat>/<id>/...

The install marker is the source-of-truth for "what version is installed":
sas-assistant reads <samples-root>/_pack-version.json after extraction, compares
against the build's expected version constant, and triggers re-download when
they differ. Putting the marker INSIDE the bundle (rather than in electron-store)
means a user-deleted samples folder is automatically detected as "not installed."

Determinism (so two builds from the same input produce byte-identical zips +
sha256):

  - File list sorted by archive path before adding
  - All entries get a fixed mtime (zero-fill epoch — predictable across machines)
  - ZIP_DEFLATED level 6 (zip default)
  - External attributes stripped to a fixed value (0o644 for files, 0o755 for dirs)

Excluded from the payload (admin convention — anything starting with "_" at
any depth):

  - _archive/, _failures/, _staging/, _scratch/
  - .DS_Store, __pycache__/, .git/
  - any path containing a "_"-prefixed segment

Usage:

  python scripts/build_pack.py --pack instruments --version 1
    # defaults: source=$SAS_OUTPUTS_DIR/instruments  out=./dist/

  python scripts/build_pack.py --pack drums --version dev \\
    --source ./test-fixtures/drums --out /tmp/

  # smoke test (writes its own fixture, builds + verifies):
  python scripts/build_pack.py --smoke-test
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PACK_REGISTRY = {
    "drums": {
        "pack_id": "sas-drum-pack",
        # Source subdir under SAS_OUTPUTS_DIR (legacy name; see Phase 0.7 rename
        # follow-up — the in-app drum-resolver still expects this folder shape
        # so the zip preserves it for now).
        "default_source_subdir": "processed",
    },
    "instruments": {
        "pack_id": "sas-instrument-pack",
        "default_source_subdir": "instruments",
    },
    "loops": {
        # The factory loop / sample library, migrated from the legacy
        # sas-sample-library-v1.dmg to a versioned zip so PackDownloadService
        # (the new pack system) can install it like the drum/instrument packs.
        "pack_id": "sas-loop-library",
        "default_source_subdir": "loops",
    },
}

# Fixed epoch for deterministic zip entry timestamps. Year 1980-01-01 is the
# zip-format minimum; using it makes mtime metadata not the source of any drift.
FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def is_excluded(path_parts: tuple[str, ...]) -> bool:
    """True if any segment in the path is excluded (starts with _ or . — except
    the leading '.' that comes from a relative-path prefix). Matches admin
    folders like _archive/, _failures/, .DS_Store, .git/, __pycache__."""
    for part in path_parts:
        if not part or part == ".":
            continue
        if part.startswith("_") or part.startswith("."):
            return True
    return False


def list_payload_files(source: Path) -> list[tuple[Path, str]]:
    """Walk `source`, return [(abs_path, archive_rel_path)] sorted by archive
    path. Excluded admin paths are filtered out. Empty dirs are not represented
    (zip auto-creates parents on add)."""
    entries: list[tuple[Path, str]] = []
    for root, dirs, files in os.walk(source):
        # Filter dirs in-place so os.walk skips excluded subtrees entirely.
        dirs[:] = sorted(d for d in dirs if not (d.startswith("_") or d.startswith(".")))
        files = sorted(files)
        for name in files:
            if name.startswith(".") or name.startswith("_"):
                continue
            abs_path = Path(root) / name
            rel_parts = abs_path.relative_to(source).parts
            if is_excluded(rel_parts):
                continue
            archive_path = "/".join(rel_parts)
            entries.append((abs_path, archive_path))
    entries.sort(key=lambda e: e[1])
    return entries


def get_git_commit() -> Optional[str]:
    """Return short HEAD sha of the cwd's git repo, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def build_version_manifest(
    pack_id: str,
    version: str,
    files: list[tuple[Path, str]],
) -> dict:
    """Build the _pack-version.json contents written into the zip root."""
    total_bytes = sum(p.stat().st_size for p, _ in files)
    return {
        "packId": pack_id,
        "version": version,
        "schemaVersion": 1,
        "buildDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sourceCommit": get_git_commit(),
        "sizeBytesUncompressed": total_bytes,
        "fileCount": len(files),
    }


def write_zip(out_path: Path, manifest: dict, files: list[tuple[Path, str]]) -> None:
    """Write a deterministic zip with manifest + payload files."""
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Manifest goes first (it's the install marker; consumers can stop
        # streaming early if they only need the version).
        info = zipfile.ZipInfo(filename="_pack-version.json", date_time=FIXED_ZIP_DATE)
        info.external_attr = 0o644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, manifest_bytes)

        for abs_path, archive_path in files:
            info = zipfile.ZipInfo(filename=archive_path, date_time=FIXED_ZIP_DATE)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(abs_path, "rb") as src:
                zf.writestr(info, src.read())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pack(pack: str, version: str, source: Path, out_dir: Path) -> Path:
    if pack not in PACK_REGISTRY:
        sys.exit(f"unknown pack {pack!r}; expected one of {sorted(PACK_REGISTRY)}")
    if not source.exists():
        sys.exit(f"source dir does not exist: {source}")
    if not source.is_dir():
        sys.exit(f"source must be a directory: {source}")

    reg = PACK_REGISTRY[pack]
    pack_id = reg["pack_id"]

    print(f"[build_pack] pack={pack} pack_id={pack_id} version={version}")
    print(f"[build_pack] source: {source}")

    files = list_payload_files(source)
    if not files:
        sys.exit(f"no payload files found under {source} (after admin-path exclusions)")

    manifest = build_version_manifest(pack_id, version, files)
    print(f"[build_pack] files: {manifest['fileCount']}  "
          f"uncompressed: {manifest['sizeBytesUncompressed']:,} bytes")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack_id}-v{version}.zip"
    if out_path.exists():
        out_path.unlink()

    t0 = time.time()
    write_zip(out_path, manifest, files)
    elapsed = time.time() - t0

    zip_size = out_path.stat().st_size
    sha = sha256_file(out_path)

    print()
    print(f"[build_pack] wrote: {out_path}")
    print(f"[build_pack] zip size:     {zip_size:,} bytes ({zip_size / 1e9:.2f} GB)")
    print(f"[build_pack] uncompressed: {manifest['sizeBytesUncompressed']:,} bytes")
    print(f"[build_pack] sha256:       {sha}")
    print(f"[build_pack] build time:   {elapsed:.1f}s")
    print()
    print("Update sas-assistant/src/shared/constants/sample-packs.ts with:")
    print(f"  expectedVersion: '{version}',")
    print(f"  sizeBytes: {zip_size},")
    print(f"  sha256: '{sha}',")
    return out_path


def smoke_test() -> None:
    """Build a tiny pack from a synthetic fixture, then verify the zip's
    contents round-trip correctly. No real samples needed."""
    print("[smoke_test] creating temp fixture...")
    with tempfile.TemporaryDirectory(prefix="build_pack_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        source = tmp / "instruments"
        dist = tmp / "dist"
        # Synthetic instrument-pack-shaped fixture:
        #   instruments/plucks/plucks-aaa/{manifest.json, sources/053.wav, zones/060.flac}
        #   instruments/_archive/should-be-excluded.txt
        #   instruments/.DS_Store (should be excluded)
        inst_dir = source / "plucks" / "plucks-aaa"
        (inst_dir / "sources").mkdir(parents=True)
        (inst_dir / "zones").mkdir(parents=True)
        (inst_dir / "manifest.json").write_text(
            json.dumps({"instrument_id": "plucks-aaa", "schema_version": 1}, indent=2)
        )
        (inst_dir / "sources" / "053.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake-wav")
        (inst_dir / "zones" / "060.flac").write_bytes(b"fLaC\x00\x00\x00\x00fake-flac")
        # Admin paths that should NOT appear in the zip:
        (source / "_archive").mkdir()
        (source / "_archive" / "should-be-excluded.txt").write_text("nope")
        (source / ".DS_Store").write_bytes(b"\x00")
        (inst_dir / "_failures").mkdir()
        (inst_dir / "_failures" / "skip-me.json").write_text("{}")

        print(f"[smoke_test] fixture at: {source}")
        out_path = build_pack(
            pack="instruments",
            version="smoke",
            source=source,
            out_dir=dist,
        )

        print("[smoke_test] verifying zip contents...")
        with zipfile.ZipFile(out_path) as zf:
            names = zf.namelist()
            print(f"[smoke_test] zip contains {len(names)} entries:")
            for n in names:
                print(f"  {n}")

            assert "_pack-version.json" in names, "missing version marker"
            manifest = json.loads(zf.read("_pack-version.json"))
            assert manifest["packId"] == "sas-instrument-pack", f"wrong packId: {manifest['packId']}"
            assert manifest["version"] == "smoke", f"wrong version: {manifest['version']}"
            assert manifest["fileCount"] == 3, f"wrong fileCount: {manifest['fileCount']} (expected 3 payload files)"

            # Excluded paths must not appear
            for n in names:
                assert "_archive" not in n, f"_archive leaked into zip: {n}"
                assert "_failures" not in n, f"_failures leaked into zip: {n}"
                assert ".DS_Store" not in n, f".DS_Store leaked into zip: {n}"

            # Expected payload paths exist
            expected = {
                "_pack-version.json",
                "plucks/plucks-aaa/manifest.json",
                "plucks/plucks-aaa/sources/053.wav",
                "plucks/plucks-aaa/zones/060.flac",
            }
            assert set(names) == expected, f"unexpected names; want {expected}, got {set(names)}"

        # Build twice and confirm sha256 stays identical (determinism check)
        print()
        print("[smoke_test] determinism check: rebuilding...")
        sha1 = sha256_file(out_path)
        out_path2 = build_pack(
            pack="instruments",
            version="smoke",
            source=source,
            out_dir=dist,
        )
        sha2 = sha256_file(out_path2)
        assert sha1 == sha2, f"non-deterministic build: {sha1} != {sha2}"
        print(f"[smoke_test] determinism OK: both builds sha256={sha1}")

        print()
        print("[smoke_test] PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pack", choices=sorted(PACK_REGISTRY.keys()),
                        help="Pack to build")
    parser.add_argument("--version", help="Pack version string (e.g. '1', 'dev', '2-rc1')")
    parser.add_argument("--source",
                        help="Source directory. Defaults to $SAS_OUTPUTS_DIR/<pack-subdir>")
    parser.add_argument("--out", default="./dist",
                        help="Output directory for the zip (default: ./dist)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a self-contained smoke test against a tiny synthetic fixture")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not args.pack or not args.version:
        parser.error("--pack and --version are required (unless --smoke-test)")

    if args.source:
        source = Path(args.source).resolve()
    else:
        sas_outputs = os.environ.get("SAS_OUTPUTS_DIR")
        if not sas_outputs:
            sys.exit("--source not given and SAS_OUTPUTS_DIR is not set")
        subdir = PACK_REGISTRY[args.pack]["default_source_subdir"]
        source = Path(sas_outputs).resolve() / subdir

    out_dir = Path(args.out).resolve()
    build_pack(pack=args.pack, version=args.version, source=source, out_dir=out_dir)


if __name__ == "__main__":
    main()
