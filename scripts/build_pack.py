"""Build a versioned, deterministic sample pack zip for distribution via GCP.

Produces:

  dist/sas-{drum,instrument}-pack-v{N}.zip

The zip contains, at its root:

  _pack-version.json   <- install marker (read by sas-app's PackDownloadService)
  <payload tree>       <- drums: drum-role folders; instruments: instruments/<cat>/<id>/...

The install marker is the source-of-truth for "what version is installed":
sas-app reads <samples-root>/_pack-version.json after extraction, compares
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

  # ready-to-consume DIRECTORY (drop its contents into the app's
  # <userData>/sample-packs/<subdir>/ — no zip/download step):
  python scripts/build_pack.py --pack instruments --version 3 --format dir

  # smoke test (writes its own fixture, builds + verifies):
  python scripts/build_pack.py --smoke-test
"""

import argparse
import hashlib
import json
import os
import shutil
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
        # Subdir the app extracts into, under <userData>/sample-packs/ — must
        # equal DRUM_PACK.installSubdir in sas-app sample-packs.ts. The --format
        # dir output is named for this so its CONTENTS drop straight in.
        "install_subdir": "drums",
    },
    "instruments": {
        "pack_id": "sas-instrument-pack",
        "default_source_subdir": "instruments",
        "install_subdir": "instruments",
    },
    "loops": {
        # The factory loop / sample library, migrated from the legacy
        # sas-sample-library-v1.dmg to a versioned zip so PackDownloadService
        # (the new pack system) can install it like the drum/instrument packs.
        "pack_id": "sas-loop-library",
        "default_source_subdir": "loops",
        "install_subdir": "loops",
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


def write_dir(out_path: Path, manifest: dict, files: list[tuple[Path, str]]) -> None:
    """Write a ready-to-consume library DIRECTORY — the exact tree the zip would
    extract to: `_pack-version.json` at the root + the payload. Its CONTENTS drop
    straight into `<userData>/sample-packs/<installSubdir>/`, so the app sees an
    installed, valid library with no zip/download step (the marker is what
    getInstalledVersion() reads)."""
    if out_path.exists():
        # Only clobber a prior build (has the marker) or an empty dir — never an
        # unknown folder we didn't create.
        if not (out_path / "_pack-version.json").exists() and any(out_path.iterdir()):
            sys.exit(f"refusing to overwrite non-pack directory: {out_path}\n"
                     f"  (no _pack-version.json and not empty — remove it yourself if intended)")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out_path / "_pack-version.json").write_bytes(manifest_bytes)
    for abs_path, archive_path in files:
        dst = out_path / archive_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, dst)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pack(pack: str, version: str, source: Path, out_dir: Path,
               fmt: str = "zip") -> dict:
    """Build the pack as a zip ('zip'), a ready-to-consume directory ('dir'), or
    both ('both'). Returns {'zip': path_or_None, 'dir': path_or_None}."""
    if pack not in PACK_REGISTRY:
        sys.exit(f"unknown pack {pack!r}; expected one of {sorted(PACK_REGISTRY)}")
    if not source.exists():
        sys.exit(f"source dir does not exist: {source}")
    if not source.is_dir():
        sys.exit(f"source must be a directory: {source}")

    reg = PACK_REGISTRY[pack]
    pack_id = reg["pack_id"]

    print(f"[build_pack] pack={pack} pack_id={pack_id} version={version} format={fmt}")
    print(f"[build_pack] source: {source}")

    files = list_payload_files(source)
    if not files:
        sys.exit(f"no payload files found under {source} (after admin-path exclusions)")

    manifest = build_version_manifest(pack_id, version, files)
    print(f"[build_pack] files: {manifest['fileCount']}  "
          f"uncompressed: {manifest['sizeBytesUncompressed']:,} bytes")

    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"zip": None, "dir": None}

    if fmt in ("zip", "both"):
        out_path = out_dir / f"{pack_id}-v{version}.zip"
        if out_path.exists():
            out_path.unlink()
        t0 = time.time()
        write_zip(out_path, manifest, files)
        elapsed = time.time() - t0
        zip_size = out_path.stat().st_size
        sha = sha256_file(out_path)
        print()
        print(f"[build_pack] wrote zip:   {out_path}")
        print(f"[build_pack] zip size:    {zip_size:,} bytes ({zip_size / 1e9:.2f} GB)")
        print(f"[build_pack] sha256:      {sha}")
        print(f"[build_pack] build time:  {elapsed:.1f}s")
        print()
        print("To publish: gsutil cp the zip to GCP, then update")
        print("sas-app/src/shared/constants/sample-packs.ts with:")
        print(f"  expectedVersion: '{version}',")
        print(f"  sizeBytes: {zip_size},")
        print(f"  sha256: '{sha}',")
        result["zip"] = out_path

    if fmt in ("dir", "both"):
        dir_path = out_dir / reg["install_subdir"]
        write_dir(dir_path, manifest, files)
        print()
        print(f"[build_pack] wrote dir:   {dir_path}/")
        print(f"[build_pack]   {manifest['fileCount']} files + _pack-version.json "
              f"(version={version})")
        print(f"[build_pack] consume out of the box — its CONTENTS go straight into the app:")
        print(f"  rsync -a '{dir_path}/' '<userData>/sample-packs/{reg['install_subdir']}/'")
        print(f"  (the app expects version '{version}' — keep it == "
              f"sas-app sample-packs.ts {pack_id} expectedVersion)")
        result["dir"] = dir_path

    return result


def _assert_zone_invariants(zones: list[dict]) -> None:
    """The instrument plugin / Tracktion sampler require zones to be disjoint,
    ordered low->high, root inside its range, and contiguous over 0..127
    (overlaps double-trigger). Mirrors the engine + enrich invariants."""
    assert zones, "no zones"
    prev_max = -1
    for z in zones:
        assert z["min_midi"] <= z["root_midi"] <= z["max_midi"], f"root outside zone: {z}"
        assert z["min_midi"] > prev_max, f"zone overlap/out-of-order: {z} (prev_max={prev_max})"
        assert z["sample"].endswith(".wav"), f"v3 zones must be WAV (mmap-able), got {z['sample']}"
        prev_max = z["max_midi"]
    assert zones[0]["min_midi"] == 0 and zones[-1]["max_midi"] == 127, "zones must cover 0..127"


def smoke_test() -> None:
    """Build a tiny pack from a synthetic fixture, then verify the zip's
    contents round-trip correctly. No real samples needed.

    The fixture is a v3 MULTI-SOURCE instrument (2 real source pitches, WAV
    zones) so this also guards the manifest invariants the instrument-resolver
    relies on."""
    print("[smoke_test] creating temp fixture...")
    with tempfile.TemporaryDirectory(prefix="build_pack_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        source = tmp / "instruments"
        dist = tmp / "dist"
        # Synthetic v3 multi-source instrument:
        #   instruments/basses/basses-aaa/
        #     manifest.json, sources/{040,052}.wav, zones/{033,040,047,052,059}.wav
        inst_dir = source / "basses" / "basses-aaa"
        (inst_dir / "sources").mkdir(parents=True)
        (inst_dir / "zones").mkdir(parents=True)
        zones = [
            {"sample": "zones/033.wav", "root_midi": 33, "min_midi": 0, "max_midi": 36},
            {"sample": "zones/040.wav", "root_midi": 40, "min_midi": 37, "max_midi": 43},
            {"sample": "zones/047.wav", "root_midi": 47, "min_midi": 44, "max_midi": 49},
            {"sample": "zones/052.wav", "root_midi": 52, "min_midi": 50, "max_midi": 55},
            {"sample": "zones/059.wav", "root_midi": 59, "min_midi": 56, "max_midi": 127},
        ]
        _assert_zone_invariants(zones)  # fixture itself must be valid
        manifest = {
            "schema_version": 1,
            "instrument_id": "basses-aaa",
            "category_id": "basses",
            "category_display": "Basses",
            "prompt": "deep analog sub bass single note",
            "open_ended": False,
            "sources": [
                {"file": "sources/040.wav", "target_pitch_midi": 40, "effective_root_midi": 40},
                {"file": "sources/052.wav", "target_pitch_midi": 52, "effective_root_midi": 52},
            ],
            "loop": None,
            "zones": zones,
            "channels": 1,
            "sample_rate": 44100,
            "bit_depth": 24,
        }
        (inst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for s in ("040", "052"):
            (inst_dir / "sources" / f"{s}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake-wav")
        for z in zones:
            (inst_dir / z["sample"]).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt fake-wav")
        # Admin paths that should NOT appear in the zip:
        (source / "_archive").mkdir()
        (source / "_archive" / "should-be-excluded.txt").write_text("nope")
        (source / ".DS_Store").write_bytes(b"\x00")
        (inst_dir / "_failures").mkdir()
        (inst_dir / "_failures" / "skip-me.json").write_text("{}")

        n_payload = 1 + 2 + len(zones)  # manifest + 2 sources + zones

        print(f"[smoke_test] fixture at: {source}")
        out_path = build_pack(
            pack="instruments",
            version="smoke",
            source=source,
            out_dir=dist,
        )["zip"]

        print("[smoke_test] verifying zip contents...")
        with zipfile.ZipFile(out_path) as zf:
            names = zf.namelist()
            print(f"[smoke_test] zip contains {len(names)} entries:")
            for n in names:
                print(f"  {n}")

            assert "_pack-version.json" in names, "missing version marker"
            vmanifest = json.loads(zf.read("_pack-version.json"))
            assert vmanifest["packId"] == "sas-instrument-pack", f"wrong packId: {vmanifest['packId']}"
            assert vmanifest["version"] == "smoke", f"wrong version: {vmanifest['version']}"
            assert vmanifest["fileCount"] == n_payload, \
                f"wrong fileCount: {vmanifest['fileCount']} (expected {n_payload})"

            # Excluded paths must not appear
            for n in names:
                assert "_archive" not in n, f"_archive leaked into zip: {n}"
                assert "_failures" not in n, f"_failures leaked into zip: {n}"
                assert ".DS_Store" not in n, f".DS_Store leaked into zip: {n}"

            # The packed manifest must still satisfy the resolver invariants.
            packed = json.loads(zf.read("basses/basses-aaa/manifest.json"))
            assert packed["schema_version"] == 1
            assert len(packed["sources"]) == 2, f"expected 2 sources, got {len(packed['sources'])}"
            _assert_zone_invariants(packed["zones"])

            expected = {"_pack-version.json", "basses/basses-aaa/manifest.json"}
            expected |= {f"basses/basses-aaa/sources/{s}.wav" for s in ("040", "052")}
            expected |= {f"basses/basses-aaa/{z['sample']}" for z in zones}
            assert set(names) == expected, f"unexpected names; want {expected}, got {set(names)}"

        # --- ready-to-consume DIRECTORY emission (--format dir) ---
        print()
        print("[smoke_test] verifying dir emission (--format dir)...")
        dres = build_pack(pack="instruments", version="smoke", source=source,
                          out_dir=dist, fmt="dir")
        dpath = dres["dir"]
        assert dpath is not None and dpath.name == "instruments", f"bad dir path: {dpath}"
        assert (dpath / "_pack-version.json").exists(), "emitted dir missing version marker"
        dv = json.loads((dpath / "_pack-version.json").read_text())
        assert dv["packId"] == "sas-instrument-pack" and dv["version"] == "smoke", dv
        emitted = {str(p.relative_to(dpath)).replace(os.sep, "/")
                   for p in dpath.rglob("*") if p.is_file()}
        # The emitted tree (marker + payload) must equal exactly the zip's entries,
        # i.e. excluded admin paths (_archive/_failures/.DS_Store) are absent too.
        assert emitted == expected, f"dir payload != zip payload; diff={emitted ^ expected}"
        print(f"[smoke_test] dir-emit OK ({len(emitted)} files incl. marker)")

        # Build twice and confirm sha256 stays identical (determinism check)
        print()
        print("[smoke_test] determinism check: rebuilding...")
        sha1 = sha256_file(out_path)
        out_path2 = build_pack(
            pack="instruments",
            version="smoke",
            source=source,
            out_dir=dist,
        )["zip"]
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
                        help="Output directory (default: ./dist)")
    parser.add_argument("--format", choices=["zip", "dir", "both"], default="zip",
                        help="zip = distributable .zip (default); dir = ready-to-consume "
                             "library dir (dist/<subdir>/) to drop into the app's "
                             "<userData>/sample-packs/<subdir>/; both")
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
    build_pack(pack=args.pack, version=args.version, source=source, out_dir=out_dir,
               fmt=args.format)


if __name__ == "__main__":
    main()
