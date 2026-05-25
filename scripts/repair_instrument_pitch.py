"""Repair mislabelled instrument zone roots in an already-built library.

Background: the gate's old crepe-median pitch measurement shipped most pitched
instruments with the wrong zone `root_midi` (frequently a perfect 4th sharp),
so the Tracktion sampler transposed every note to the wrong pitch — instruments
sounded out of key. `gate_pitched.spectral_fundamental_midi` fixes the detector
going forward; this script repairs the EXISTING library WITHOUT regenerating any
audio from the model.

It re-derives each instrument from its pristine `sources/*.wav` (the gated raw
winner), exactly as `enrich_pitched.py` does, but using the fixed detector:
re-detect → re-correct → normalize → (trim) → re-render zones → rewrite manifest.
Re-deriving from the source (not the existing zones) matters: the old bad
correction baked a sub-semitone error into the shipped zones.

Safe by default: DRY-RUN prints the before/after root for every instrument and
writes nothing. `--apply` re-renders zones and rewrites manifests, backing up
each manifest to `manifest.json.bak` first.

    # report what would change across the whole library (writes nothing):
    python scripts/repair_instrument_pitch.py --root "<samples>/instruments"

    # one category, or one instrument:
    python scripts/repair_instrument_pitch.py --root <...> --category synths
    python scripts/repair_instrument_pitch.py --root <...> --instrument synths-8ee8f262

    # actually fix the files (re-renders zones via rubberband):
    python scripts/repair_instrument_pitch.py --root <...> --instrument synths-8ee8f262 --apply
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_pitched import spectral_fundamental_midi, hz_to_midi, midi_to_hz  # noqa: E402
from enrich_pitched import (  # noqa: E402
    pitch_shift,
    normalize_lufs,
    trim_to_sustain,
    midi_to_filename,
)
from pitched_category_config import PITCHED_CATEGORIES  # noqa: E402

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note(midi: float) -> str:
    if midi is None or not math.isfinite(midi):
        return "?"
    m = int(round(midi))
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def _effective_root(measured_midi: float, target_midi: int, max_correction: int) -> int:
    """Mirror enrich_pitched's correction policy: within max_correction of the
    target → lock to target; else snap to the nearest integer semitone."""
    if abs(measured_midi - target_midi) <= max_correction:
        return target_midi
    return int(round(measured_midi))


def _build_zone_layout(effective_root: int, span: int, step: int) -> list[int]:
    roots: list[int] = []
    for delta in range(-span, span + 1, step):
        r = effective_root + delta
        if 0 <= r <= 127:
            roots.append(r)
    if effective_root not in roots:
        roots.append(effective_root)
        roots.sort()
    return roots


def repair_instrument(inst_dir: Path, apply: bool) -> dict:
    """Returns a report dict; mutates files only when apply=True."""
    manifest_path = inst_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cat_id = manifest.get("category_id")
    cfg = PITCHED_CATEGORIES.get(cat_id)
    sources = manifest.get("sources") or []
    if not sources:
        return {"id": inst_dir.name, "status": "skip", "reason": "no sources"}
    if cfg is not None and cfg.skip_pitch_shift:
        return {"id": inst_dir.name, "status": "skip", "reason": "fx (no pitch)"}

    src0 = sources[0]
    target = src0.get("target_pitch_midi")
    if target is None:
        return {"id": inst_dir.name, "status": "skip", "reason": "no target pitch"}
    src_path = inst_dir / src0.get("file", "")
    if not src_path.exists():
        return {"id": inst_dir.name, "status": "skip", "reason": f"source missing ({src0.get('file')})"}

    max_corr = getattr(cfg, "max_correction_semitones", 3) if cfg else 3
    span = cfg.zone_span_semitones if cfg else 12
    step = cfg.zone_step_semitones if cfg else 3

    y2d, sr = sf.read(str(src_path), always_2d=True)
    mono = y2d.mean(axis=1)
    measured = spectral_fundamental_midi(mono, sr, target)
    old_measured = src0.get("measured_pitch_midi")
    # The instrument's CURRENT root is whichever zone was the unshifted source —
    # i.e. the zone whose root equals the old effective root. Recover it from the
    # old measured pitch for reporting.
    old_root = (_effective_root(float(old_measured), target, max_corr)
                if old_measured is not None and math.isfinite(float(old_measured)) else None)

    if not math.isfinite(measured):
        return {"id": inst_dir.name, "status": "skip", "reason": "no clear pitch"}

    new_root = _effective_root(measured, target, max_corr)
    report = {
        "id": inst_dir.name, "status": "ok", "target": target,
        "old_measured": old_measured, "old_root": old_root,
        "new_measured": round(measured, 2), "new_root": new_root,
        "delta": (new_root - old_root) if old_root is not None else None,
    }
    if not apply:
        return report

    # ----- re-derive audio exactly like enrich_pitched, with the fixed pitch ---
    shift = new_root - measured
    y = pitch_shift(y2d, sr, shift, preserve_formants=True) if abs(shift) > 0.05 else y2d.copy()
    correction_cents = int(round(shift * 100.0))
    if cfg is not None and cfg.open_ended:
        y = trim_to_sustain(y, sr, cfg.min_sustain_seconds, pad_seconds=0.5)
    y_norm, _ = normalize_lufs(y, sr)

    zones_dir = inst_dir / "zones"
    roots = _build_zone_layout(new_root, span, step)
    zones = []
    new_files: set[str] = set()
    # Render the NEW zone set first (mostly different filenames since they're
    # named by root). Old zones stay in place until the manifest is rewritten,
    # so a crash mid-render leaves the instrument working on its old zones.
    for i, root in enumerate(roots):
        min_midi = 0 if i == 0 else (roots[i - 1] + root) // 2 + 1
        max_midi = 127 if i == len(roots) - 1 else (root + roots[i + 1]) // 2
        fname = f"{midi_to_filename(root)}.flac"
        new_files.add(fname)
        if root == new_root:
            sf.write(str(zones_dir / fname), y_norm, sr, format="FLAC", subtype="PCM_24")
        else:
            shifted = pitch_shift(y_norm, sr, semitones=root - new_root, preserve_formants=True)
            sf.write(str(zones_dir / fname), shifted, sr, format="FLAC", subtype="PCM_24")
        zones.append({"sample": f"zones/{fname}", "root_midi": root,
                      "min_midi": min_midi, "max_midi": max_midi})

    # ----- rewrite manifest (backup first) -----
    shutil.copy2(manifest_path, manifest_path.with_suffix(".json.bak"))
    src0["measured_pitch_midi"] = round(measured, 2)
    src0["measured_pitch_cents_offset"] = round((measured - target) * 100.0, 1)
    src0["pitch_correction_applied_cents"] = correction_cents
    manifest["zones"] = zones
    manifest["pitch_repaired"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Prune stale zone files no longer referenced by the new manifest.
    for old in zones_dir.glob("*.flac"):
        if old.name not in new_files:
            old.unlink()
    report["status"] = "applied"
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="instruments/ root (contains <category>/<id>/manifest.json)")
    ap.add_argument("--category", default=None, help="only this category")
    ap.add_argument("--instrument", default=None, help="only this instrument id")
    ap.add_argument("--apply", action="store_true", help="re-render + rewrite (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N instruments (testing)")
    args = ap.parse_args()

    manifests = sorted(glob.glob(os.path.join(args.root, "*", "*", "manifest.json")))
    reports = []
    changed = octave_only = classfix = 0
    for mp in manifests:
        inst_dir = Path(mp).parent
        if args.category and inst_dir.parent.name != args.category:
            continue
        if args.instrument and inst_dir.name != args.instrument:
            continue
        r = repair_instrument(inst_dir, apply=args.apply)
        reports.append(r)
        if r.get("status") in ("ok", "applied") and r.get("delta") not in (None, 0):
            changed += 1
            if r["delta"] % 12 == 0:
                octave_only += 1
            else:
                classfix += 1
        if args.limit and len(reports) >= args.limit:
            break

    mode = "APPLIED" if args.apply else "DRY-RUN (no files written)"
    print(f"\n=== Instrument pitch repair — {mode} ===")
    for r in reports:
        if r["status"] == "skip":
            continue
        tag = "" if r.get("delta") in (None, 0) else f"  Δ={r['delta']:+d} st"
        flag = "  *** PITCH-CLASS FIX (was out of key)" if r.get("delta") not in (None, 0) and r["delta"] % 12 != 0 else ""
        print(f"  {r['id']:22s} target={_note(r['target']):4s} "
              f"OLD root={_note(r['old_root']):5s} -> NEW root={_note(r['new_root']):5s}"
              f" (measured {r['new_measured']}){tag}{flag}")
    skipped = sum(1 for r in reports if r["status"] == "skip")
    print(f"\n  {len(reports) - skipped} instrument(s) evaluated, {skipped} skipped.")
    print(f"  {changed} would change root  ({classfix} pitch-CLASS fixes / out-of-key, "
          f"{octave_only} octave-only).")
    if not args.apply and changed:
        print("  Re-run with --apply to re-render zones + rewrite manifests (backs up each manifest).")


if __name__ == "__main__":
    main()
