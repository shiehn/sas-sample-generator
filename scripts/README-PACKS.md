# Sample pack release runbook

This is the operator handbook for cutting a new versioned sample-pack zip
and publishing it for the sas-app Electron app to pick up.

There are two independent packs:

| Pack | Source dir | Distributed as | Approx size |
|------|------------|----------------|-------------|
| Drums | `${SAS_OUTPUTS_DIR}/processed/` | `sas-drum-pack-v{N}.zip` | ~2–3 GB (v3, 24 roles) |
| Instruments | `${SAS_OUTPUTS_DIR}/instruments/` | `sas-instrument-pack-v{N}.zip` | ~20–24 GB (v3, 28 cats × ~150) |

Both publish to the same GCP bucket. The app downloads each on demand from
its plugin panel, with a hardcoded expected-version per app build.

> **v3 changes (this run).** Zones are **16-bit WAV**, not 24-bit FLAC: Tracktion's
> SamplerPlugin CPU-decodes FLAC on the message thread (load stall), but mmaps WAV
> instantly. Sources stay 24-bit WAV. Multi-source instruments carry 2–4
> `sources/<root>.wav`. The category set was restructured (28 pitched, 24 drum;
> pitched `fx` retired → one-shots), so **both packs bump a major version** and
> the in-app constants must update in lockstep (step 3). Never reuse a published
> version number.

---

## 1. Build the pack

```bash
cd sas-sample-generator

# Smoke-test the build script first (cheap, no real samples):
python scripts/build_pack.py --smoke-test

# Real build — drums:
python scripts/build_pack.py --pack drums --version 1

# Real build — instruments (defaults to $SAS_OUTPUTS_DIR/instruments):
python scripts/build_pack.py --pack instruments --version 1

# Override source path explicitly (e.g. samples already moved to user-data):
python scripts/build_pack.py --pack instruments --version 1 \
    --source ~/Library/Application\ Support/signals-and-sorcery/sample-packs/instruments

# Skip the zip — emit a ready-to-consume DIRECTORY (dist/<subdir>/ + the
# _pack-version.json marker) to rsync straight into the app's install dir,
# no download step:
python scripts/build_pack.py --pack instruments --version 1 --format dir
# both libraries at once: DRUM_VERSION=N INSTRUMENT_VERSION=N ./scripts/build_libraries.sh
```

The script prints, at the end:

```
[build_pack] wrote: ./dist/sas-instrument-pack-v1.zip
[build_pack] zip size:     11_534_336 bytes (~10.7 GB)
[build_pack] sha256:       3e2a4f...
```

Copy the size + sha256. You will paste them into the app constants in step 3.

---

## 2. Upload to GCP

```bash
gsutil cp dist/sas-drum-pack-v1.zip gs://docs-assets/
gsutil cp dist/sas-instrument-pack-v1.zip gs://docs-assets/
```

The bucket is `gs://docs-assets/` (same bucket the existing loops library
pack uses). Files should be publicly readable — check with:

```bash
curl -I https://storage.googleapis.com/docs-assets/sas-drum-pack-v1.zip
# Expect 200 OK with content-length matching the zip size
```

---

## 3. Wire the version into the app build

Edit `sas-app/src/shared/constants/sample-packs.ts`:

```typescript
export const DRUM_PACK: PackConfig = {
  packId: 'sas-drum-pack',
  expectedVersion: '1',                            // <- bump per release
  installSubdir: 'drums',
  downloadUrl: `${PACK_BASE_URL}/sas-drum-pack-v1.zip`,  // <- bump version in URL
  sizeBytes: 1_534_336_512,                        // <- paste from build script
  sha256: '3e2a4f...',                             // <- paste from build script
  ...
};
```

Commit + push. The next sas-app build will:

1. On launch / plugin activate, read `<userData>/sample-packs/drums/_pack-version.json`
2. If the marker is missing → show "Sample library not installed" CTA
3. If the marker version differs from `expectedVersion` → show "Update available" CTA
4. If the user clicks Download → fetch from GCP, verify size, extract,
   atomic-rename into place, marker updates

End users don't need to do anything — re-opening the app post-update
detects the version mismatch and prompts a re-download in the plugin panel.

---

## What's IN the zip

```
sas-instrument-pack-v1.zip
├── _pack-version.json         <- install marker (the source of truth on disk)
├── plucks/
│   ├── plucks-aaa/
│   │   ├── manifest.json
│   │   ├── sources/
│   │   │   └── 060.wav        <- 24-bit; multi-source instruments have 2–4 here
│   │   └── zones/
│   │       └── 060.wav        <- 16-bit WAV (mmap-able; v3 — was .flac)
│   └── ...
├── basses/
│   └── basses-aaa/
│       ├── sources/{028,040,052}.wav   <- 3 real source pitches (multi-source)
│       └── zones/...wav
└── ... (~28 pitched categories; flat role folders for ~24 drum roles)
```

`_pack-version.json`:

```json
{
  "packId": "sas-instrument-pack",
  "version": "1",
  "schemaVersion": 1,
  "buildDate": "2026-05-23T00:00:00Z",
  "sourceCommit": "abc1234",
  "sizeBytesUncompressed": 11653304575,
  "fileCount": 21333
}
```

---

## What's EXCLUDED from the zip

Any folder or file whose name starts with `_` or `.`:

- `_archive/` (old samples we keep but don't ship)
- `_failures/` (gate-rejected raws)
- `.DS_Store`, `.git/`, `__pycache__/`

This is enforced in `build_pack.py:is_excluded`. To intentionally ship
something whose name starts with `_`, rename it first.

---

## Determinism

`build_pack.py` uses:

- Sorted file list
- Fixed mtime (1980-01-01) on every zip entry
- Fixed external attrs (0o644)
- ZIP_DEFLATED level 6

Two builds from byte-identical source trees produce byte-identical zips with
matching sha256. The smoke test verifies this (`--smoke-test` rebuilds and
asserts sha256 equality).

---

## Versioning rules

- Use plain integer strings: `"1"`, `"2"`, `"3"`. No dots.
- The version number bumps independently per pack. A drum-pack v2 does not
  imply or require an instrument-pack v2.
- Once a version is uploaded to GCP, **never overwrite**. Bump to a new
  version instead. The installed-version check against the marker is what
  signals "you need to re-download" — overwriting would silently corrupt
  existing installs.
- `dev` and other non-integer versions are reserved for local development;
  don't ship them.
