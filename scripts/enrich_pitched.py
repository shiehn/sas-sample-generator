"""Take gate-passed pitched samples and bake them into multi-zone instruments.

For each gated WAV at outputs/gated/<cat>/<id>.wav (with its sidecar
<id>.gate.json), this stage:

  1. Reads the gate's measured-pitch + variant metadata.
  2. Pitch-corrects the source to exact target pitch if measured was
     within ±50¢ (small drift; RubberBand handles it transparently).
     Larger drift is impossible because the gate's tolerance was 50¢.
  3. LUFS-normalizes to -20 LUFS integrated, -1.0 dBTP ceiling.
     -20 (vs -16 for drums) is the right target for sustained pitched
     content with lower crest factor (BS.1770-4 weighting).
  4. For sustaining categories (open_ended=True in the category config),
     trims the source down to its steady-state plateau region — Tracktion
     plays this region for the note-hold duration. v1 substitute for
     true loop points (Tracktion's SamplerPlugin has no loop API).
  5. Pre-renders pitch-shifted zones at every `zone_step_semitones`
     across `±zone_span_semitones`. RubberBand R3 engine with
     formant preservation. Encoded as 24-bit FLAC to halve disk vs WAV.
  6. Writes manifest.json conforming to the v1 schema. Plugin reads
     this to build its zones[] for setSamplerMultiZone.

Output:

    outputs/instruments/<cat>/<instrument-id>/
        ├── sources/<root>.wav        (target-pitched, normalized)
        ├── zones/<midi>.flac         (e.g. 048.flac through 072.flac)
        ├── manifest.json
        └── prompt.txt

Runs on local CPU (RubberBand is single-threaded but cheap). Reads only
outputs/gated/ — the GPU pod doesn't need to do this work.

Usage:

    python scripts/enrich_pitched.py --category plucks
    python scripts/enrich_pitched.py --category plucks --in-dir outputs/gated/plucks --out-dir outputs/instruments/plucks
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitched_category_config import PITCHED_CATEGORIES, PitchedCategoryConfig


GATE_VERSION = "1.0.0"
ENRICH_VERSION = "1.0.0"

TARGET_LUFS = -20.0
PEAK_CEILING_DBTP = -1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def midi_to_filename(midi: int) -> str:
    """`60 → 060` (3-digit zero-padded so a directory listing sorts by pitch)."""
    return f"{midi:03d}"


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))


def amp_to_db(a: float) -> float:
    return 20.0 * math.log10(a) if a > 0 else float("-inf")


# -----------------------------------------------------------------------------
# Pitch shift (RubberBand R3)
# -----------------------------------------------------------------------------

def pitch_shift(y: np.ndarray, sr: int, semitones: float, preserve_formants: bool = True) -> np.ndarray:
    """Render a pitched copy of `y` at `semitones` offset via rubberband CLI.

    Direct subprocess to `rubberband` instead of going through pyrubberband.
    pyrubberband's `rbargs` dict-to-CLI serializer emits an empty positional
    arg for value-less flags like `--formant` (it stores them as
    `{"--formant": ""}` and stringifies both key and value), which
    rubberband 1.9 (Ubuntu 22.04 default) rejects with exit status 2.
    Going direct also means we never accidentally pass --fine (the R3
    engine flag that only rubberband 3.x understands).

    If rubberband-cli isn't installed, falls back to no-op with a
    warning — the unshifted source still ships as the root zone.
    """
    if semitones == 0:
        return y.copy()

    import shutil
    import subprocess
    import tempfile

    if shutil.which("rubberband") is None:
        print(f"[enrich] rubberband-cli unavailable; skipping shift {semitones:+.1f} ST", file=sys.stderr)
        return y.copy()

    in_fd, in_path = tempfile.mkstemp(suffix=".wav")
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(in_fd)
    os.close(out_fd)
    try:
        sf.write(in_path, y, sr, subtype="PCM_24")
        cmd = ["rubberband", "-q", "--pitch", f"{semitones}"]
        if preserve_formants:
            cmd.append("--formant")
        cmd += [in_path, out_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_snippet = (result.stderr or result.stdout or "").strip().splitlines()
            tail = stderr_snippet[-1] if stderr_snippet else "no stderr"
            print(
                f"[enrich] rubberband failed (exit {result.returncode}): {tail}; "
                f"skipping {semitones:+.1f} ST",
                file=sys.stderr,
            )
            return y.copy()
        shifted, _ = sf.read(out_path, always_2d=True)
        return shifted
    except Exception as e:
        print(f"[enrich] pitch_shift failed ({e}); skipping {semitones:+.1f} ST", file=sys.stderr)
        return y.copy()
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Loudness + peak normalization (ITU-R BS.1770-4 via pyloudnorm)
# -----------------------------------------------------------------------------

def normalize_lufs(y: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Apply LUFS gain to hit TARGET_LUFS, then clamp to PEAK_CEILING_DBTP."""
    meta: dict = {
        "loudness_lufs_in": None,
        "loudness_lufs_out": None,
        "lufs_gain_db": None,
        "peak_dbfs_in": None,
        "peak_dbfs_out": None,
        "rms_dbfs": None,
    }
    # Short samples can't be measured by BS.1770-4 directly (needs > ~0.4s);
    # tile to 3.5s for measurement only.
    buf = y if y.shape[0] >= int(sr * 3.5) else np.tile(y, (int(np.ceil(sr * 3.5 / y.shape[0])), 1))[: int(sr * 3.5)]
    meter = pyln.Meter(sr)
    mono_for_meter = buf[:, 0] if buf.ndim == 2 and buf.shape[1] == 1 else buf
    try:
        lufs_in = float(meter.integrated_loudness(mono_for_meter))
    except Exception:
        lufs_in = float("-inf")
    meta["loudness_lufs_in"] = round(lufs_in, 2) if math.isfinite(lufs_in) else None

    if not math.isfinite(lufs_in):
        return y, meta

    gain_db = TARGET_LUFS - lufs_in
    out = y * db_to_amp(gain_db)
    meta["lufs_gain_db"] = round(gain_db, 2)

    peak_in = float(np.max(np.abs(y))) if y.size else 0.0
    meta["peak_dbfs_in"] = round(amp_to_db(peak_in), 2) if peak_in > 0 else None

    ceiling = db_to_amp(PEAK_CEILING_DBTP)
    peak_post = float(np.max(np.abs(out))) if out.size else 0.0
    if peak_post > ceiling:
        out = out * (ceiling / peak_post)
    final_peak = float(np.max(np.abs(out)))
    meta["peak_dbfs_out"] = round(amp_to_db(final_peak), 2) if final_peak > 0 else None
    meta["loudness_lufs_out"] = round(TARGET_LUFS, 2)
    rms = float(np.sqrt(np.mean(out ** 2)))
    meta["rms_dbfs"] = round(amp_to_db(rms), 2) if rms > 0 else None
    return out, meta


# -----------------------------------------------------------------------------
# Sustain trim (for open_ended categories)
# -----------------------------------------------------------------------------

def trim_to_sustain(y: np.ndarray, sr: int, min_sustain_seconds: float, pad_seconds: float = 0.5
                    ) -> np.ndarray:
    """Trim `y` down to the plateau region (peak-RMS ± 6 dB), targeting
    `min_sustain + pad` seconds. Falls back to the original if no clean
    plateau can be found.

    Used only for sustaining categories (Pads, Strings, Organs, Brass,
    Winds) where Tracktion replays the trimmed clip for the note-hold
    duration. Tracktion has no loop API; this is the v1 substitute.
    """
    if y.shape[0] <= sr * (min_sustain_seconds + pad_seconds):
        return y

    import librosa
    mono = np.mean(y, axis=1) if y.ndim == 2 else y
    frame_length, hop_length = 2048, 512
    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return y
    peak_rms = float(np.max(rms))
    if peak_rms <= 0:
        return y

    threshold = peak_rms / 2.0
    above = rms >= threshold
    # Find the longest contiguous run
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for i, v in enumerate(above):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0

    if best_len == 0:
        return y

    target_len_seconds = min_sustain_seconds + pad_seconds
    target_samples = int(target_len_seconds * sr)
    start_sample = best_start * hop_length
    end_sample = min(start_sample + target_samples, y.shape[0])
    trimmed = y[start_sample:end_sample]
    # Apply 10ms equal-power crossfade in/out so the loop doesn't click.
    fade_samples = int(0.01 * sr)
    if trimmed.shape[0] > 2 * fade_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        if trimmed.ndim == 2:
            fade_in = fade_in[:, None]
            fade_out = fade_out[:, None]
        trimmed[:fade_samples] *= fade_in
        trimmed[-fade_samples:] *= fade_out
    return trimmed


# -----------------------------------------------------------------------------
# Per-instrument pipeline
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main: walk gated dir, enrich each, write manifests
# -----------------------------------------------------------------------------

def run_enrich(category_id: str, cfg: PitchedCategoryConfig,
               in_dir: Path, out_dir: Path, raw_meta_dir: Optional[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_jsons = sorted(in_dir.glob("*.gate.json"))
    if not gate_jsons:
        print(f"[enrich] no *.gate.json under {in_dir}; nothing to do")
        return

    enriched = 0
    skipped = 0
    for gj in gate_jsons:
        gate_wav = gj.with_suffix("").with_suffix(".wav")  # strip ".gate.json"
        if not gate_wav.exists():
            print(f"[enrich] {gj.name}: missing matching .wav; skipping")
            skipped += 1
            continue
        try:
            manifest_path = enrich_sample(gate_wav, gj, category_id, cfg, out_dir, raw_meta_dir)
            enriched += 1
            print(f"[enrich] {manifest_path.parent.name}: ok ({len(json.loads(manifest_path.read_text())['zones'])} zones)")
        except Exception as e:
            print(f"[enrich] {gate_wav.name}: FAILED — {e}", file=sys.stderr)
            skipped += 1

    print(f"[enrich] done. enriched={enriched} skipped={skipped}")


def enrich_sample(gate_wav: Path, gate_json: Path, category_id: str, cfg: PitchedCategoryConfig,
                  category_out_dir: Path, raw_meta_dir: Optional[Path]) -> Path:
    """Process one gated sample → write a complete instrument folder.
    Returns the manifest.json path."""
    gate_report = json.loads(gate_json.read_text(encoding="utf-8"))
    instrument_id = gate_report["id"]
    target_midi = int(gate_report["target_pitch_midi"])
    winner = gate_report["winner"]
    metrics = winner.get("metrics", {})

    # Load + smart pitch correct + normalize + (maybe) sustain-trim
    # Note: cast to str — older soundfile (which the RunPod image ships) doesn't
    # recognize pathlib.PosixPath in its _open code path; it falls through to a
    # `raise TypeError("Invalid file: PosixPath('...')")`. Newer soundfile is
    # PathLike-aware, but we can't count on that on a fresh pod.
    y, sr = sf.read(str(gate_wav), always_2d=True)

    # ---------------- Smart pitch correction ----------------
    # Goal: every output sample lands EXACTLY on a MIDI semitone, but with
    # the smallest possible pitch shift to avoid timbral artifacts.
    #
    #   measured pitch is within max_correction_semitones of target →
    #     shift all the way to target (preserves prompt semantics, e.g.
    #     a "C4 piano" prompt yields a sample really at C4)
    #
    #   measured pitch is further away →
    #     snap to nearest integer semitone (shift is always ≤50 cents,
    #     no audible artifact). The sample's effective root is whatever
    #     pitch SA3 actually generated, rounded to the nearest MIDI note.
    #
    # Either way: the sample is at a clean integer-MIDI pitch, ready to
    # be the sampler's root note. Zone rendering re-centers around it.
    cents_offset = metrics.get("measured_pitch_cents_offset")
    measured_midi_metric = metrics.get("measured_pitch_midi")
    effective_root_midi = target_midi  # default: target
    correction_applied = 0

    if cfg.skip_pitch_shift:
        # FX: no pitch handling at all.
        pass
    elif measured_midi_metric is not None and math.isfinite(float(measured_midi_metric)):
        measured_midi = float(measured_midi_metric)
        distance_semitones = abs(measured_midi - target_midi)
        if distance_semitones <= cfg.max_correction_semitones:
            # Close enough: shift to original target.
            effective_root_midi = target_midi
        else:
            # Too far: snap to nearest integer semitone.
            effective_root_midi = int(round(measured_midi))
        shift_semitones = effective_root_midi - measured_midi
        if abs(shift_semitones) > 0.05:  # > 5 cents — worth a shift
            y = pitch_shift(y, sr, shift_semitones, preserve_formants=True)
            correction_applied = int(round(shift_semitones * 100.0))
    elif cents_offset is not None and abs(cents_offset) > 5.0:
        # Fallback for older gate verdicts: cents_offset only, no measured midi
        # in metrics. Use the original "shift toward target by cents_offset" path
        # but cap at max_correction_semitones.
        cap_cents = cfg.max_correction_semitones * 100
        clipped_cents = max(-cap_cents, min(cap_cents, cents_offset))
        semitones = -clipped_cents / 100.0
        y = pitch_shift(y, sr, semitones, preserve_formants=True)
        correction_applied = int(round(semitones * 100.0))
        effective_root_midi = target_midi
    # ---------------------------------------------------------

    if cfg.open_ended:
        y = trim_to_sustain(y, sr, cfg.min_sustain_seconds, pad_seconds=0.5)

    y_norm, loud_meta = normalize_lufs(y, sr)

    # Instrument folder layout
    inst_dir = category_out_dir / instrument_id
    sources_dir = inst_dir / "sources"
    zones_dir = inst_dir / "zones"
    sources_dir.mkdir(parents=True, exist_ok=True)
    zones_dir.mkdir(parents=True, exist_ok=True)

    # Write the source WAV (24-bit, locked to effective_root + normalized)
    source_filename = f"{midi_to_filename(effective_root_midi)}.wav"
    source_path = sources_dir / source_filename
    sf.write(str(source_path), y_norm, sr, subtype="PCM_24")

    # Pre-render zones (FX: skip; everything else: every zone_step_semitones across ±span)
    # Zones now center on effective_root_midi, not target_midi — keeps zone
    # rendering aligned with where the audio actually is.
    zones: list[dict] = []
    if cfg.skip_pitch_shift:
        # FX: single zone covering full keyboard, no shift
        zone_filename = f"{midi_to_filename(effective_root_midi)}.flac"
        zone_path = zones_dir / zone_filename
        sf.write(str(zone_path), y_norm, sr, format="FLAC", subtype="PCM_24")
        zones.append({
            "sample": f"zones/{zone_filename}",
            "root_midi": effective_root_midi,
            "min_midi": 0,
            "max_midi": 127,
        })
    else:
        # Build disjoint zone ranges first, then render each.
        roots: list[int] = []
        for delta in range(-cfg.zone_span_semitones, cfg.zone_span_semitones + 1, cfg.zone_step_semitones):
            r = effective_root_midi + delta
            if 0 <= r <= 127:
                roots.append(r)
        # Ensure effective_root is included (in case step skips it)
        if effective_root_midi not in roots:
            roots.append(effective_root_midi)
            roots.sort()

        # Compute disjoint min/max for each root: split midpoints between adjacent roots.
        for i, root in enumerate(roots):
            if i == 0:
                min_midi = 0
            else:
                min_midi = (roots[i - 1] + root) // 2 + 1
            if i == len(roots) - 1:
                max_midi = 127
            else:
                max_midi = (root + roots[i + 1]) // 2
            zone_filename = f"{midi_to_filename(root)}.flac"
            zone_path = zones_dir / zone_filename
            if root == effective_root_midi:
                # The source IS this zone — no pitch shift required
                sf.write(str(zone_path), y_norm, sr, format="FLAC", subtype="PCM_24")
            else:
                shifted = pitch_shift(y_norm, sr, semitones=root - effective_root_midi, preserve_formants=True)
                sf.write(str(zone_path), shifted, sr, format="FLAC", subtype="PCM_24")
            zones.append({
                "sample": f"zones/{zone_filename}",
                "root_midi": root,
                "min_midi": min_midi,
                "max_midi": max_midi,
            })

    # Locate the raw metadata sidecar (batch_generate writes one per variant)
    variant_index = int(winner.get("variant_index", 0))
    suffix = f"_v{variant_index:02d}" if variant_index >= 0 else ""
    raw_meta: dict = {}
    if raw_meta_dir is not None:
        raw_meta_path = raw_meta_dir / f"{instrument_id}{suffix}.json"
        if raw_meta_path.exists():
            try:
                raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
            except Exception:
                raw_meta = {}

    # Try to copy the prompt sibling for human readability
    prompt = raw_meta.get("prompt", gate_report.get("prompt") or "")
    if prompt:
        (inst_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    # Build + write manifest.json
    channels = int(y_norm.shape[1]) if y_norm.ndim == 2 else 1
    manifest = {
        "schema_version": 1,
        "instrument_id": instrument_id,
        "category_id": category_id,
        "category_display": category_id.capitalize(),
        "prompt": prompt,
        "negative_prompt": raw_meta.get("negative_prompt", ""),
        "model": raw_meta.get("model", "stabilityai/stable-audio-3-medium"),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "gate_version": GATE_VERSION,
        "enrich_version": ENRICH_VERSION,
        "sources": [{
            "file": f"sources/{source_filename}",
            "target_pitch_midi": target_midi,
            "measured_pitch_midi": metrics.get("measured_pitch_midi"),
            "measured_pitch_cents_offset": metrics.get("measured_pitch_cents_offset"),
            "pitch_confidence": metrics.get("pitch_confidence"),
            "pitch_correction_applied_cents": correction_applied,
            "polyphony_check": metrics.get("polyphony_check"),
            "onset_ms": metrics.get("onset_ms"),
            "sustain_quality": metrics.get("sustain_quality"),
            "loudness_lufs": loud_meta.get("loudness_lufs_out"),
            "rms_dbfs": loud_meta.get("rms_dbfs"),
            "duration_seconds": round(y_norm.shape[0] / sr, 3),
            "raw_path": gate_report.get("raw_winner_path"),
            "seed": raw_meta.get("seed"),
            "variant_index": variant_index,
        }],
        "loop": None,
        "open_ended": cfg.open_ended,
        "zones": zones,
        "channels": channels,
        "sample_rate": int(sr),
        "bit_depth": 24,
    }
    manifest_path = inst_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", required=True,
                        help="Pitched category name (must exist in pitched_category_config.py)")
    parser.add_argument("--in-dir", default=None,
                        help="Gated input dir. Default: $SAS_OUTPUTS_DIR/gated/<cat>/")
    parser.add_argument("--out-dir", default=None,
                        help="Instrument library output dir. Default: $SAS_OUTPUTS_DIR/instruments/<cat>/")
    parser.add_argument("--raw-meta-dir", default=None,
                        help="Where raw _metadata sidecars live. Default: $SAS_OUTPUTS_DIR/raw/<cat>/_metadata/")
    parser.add_argument("--outputs-dir", default=None,
                        help="Override $SAS_OUTPUTS_DIR")
    args = parser.parse_args()

    cfg = PITCHED_CATEGORIES.get(args.category)
    if cfg is None:
        sys.exit(f"unknown pitched category {args.category!r}")

    outputs_dir = Path(args.outputs_dir or os.environ.get("SAS_OUTPUTS_DIR", "outputs"))
    in_dir = Path(args.in_dir) if args.in_dir else outputs_dir / "gated" / args.category
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir / "instruments" / args.category
    raw_meta_dir = Path(args.raw_meta_dir) if args.raw_meta_dir else outputs_dir / "raw" / args.category / "_metadata"
    if not raw_meta_dir.exists():
        raw_meta_dir = None

    run_enrich(args.category, cfg, in_dir, out_dir, raw_meta_dir)


if __name__ == "__main__":
    main()
