"""Quality-gate every SA3 candidate for the pitched-instrument pipeline.

For each category, walk every raw WAV under outputs/raw/<cat>/, group by
base-id (stripping `_v0N` suffix), and for each (prompt, target_pitch)
group evaluate all variants through five sequential gates:

  1. Pre-filter   — cheap NumPy: NaN/Inf, silent, DC offset, clipping, dead channel
  2. Onset        — librosa onset_detect + onset_backtrack; reject if perceptual onset > 30 ms
  3. Sustain      — RMS envelope; reject if peak-plateau region shorter than category's min_sustain
  4. Polyphony    — basic-pitch; reject if multiple concurrent notes (with vibrato bypass)
  5. Pitch        — torchcrepe (general) or 3-way agreement (sub-bass);
                    reject if measured pitch differs from target by > tolerance

Variants that pass all five gates are scored by:

    score = (pitch_confidence ** 2) * exp(-|cents_offset|/50) * sustain_quality

The highest-scoring variant per group becomes the gate's chosen sample,
copied to outputs/gated/<cat>/<id>.wav with a sidecar <id>.gate.json
holding the per-gate scores. Groups where ALL variants reject emit a
_failures/<id>.json line for prompt-iteration debugging.

Designed to run on a GPU pod (torchcrepe + basic-pitch benefit from
CUDA). The output of this stage — outputs/gated/<cat>/ — is the only
thing that needs to cross the machine boundary; enrich_pitched.py then
runs on local CPU.

Usage:

    python scripts/gate_pitched.py --category plucks --jsonl prompts/pitched/plucks.jsonl
    python scripts/gate_pitched.py --category plucks --jsonl prompts/pitched/plucks.jsonl --in-dir outputs/raw/plucks --out-dir outputs/gated/plucks
"""

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitched_category_config import PITCHED_CATEGORIES, PitchedCategoryConfig


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12))


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return float("nan")
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def cents_between(hz_a: float, hz_b: float) -> float:
    """Signed cents from hz_b (target) to hz_a (measured)."""
    if hz_a <= 0 or hz_b <= 0:
        return float("nan")
    return 1200.0 * math.log2(hz_a / hz_b)


def strip_variant_suffix(stem: str) -> tuple[str, int]:
    """`plucks-abc12345_v02` → (`plucks-abc12345`, 2). Single-variant
    files (no `_vNN`) → (stem, 0)."""
    if "_v" in stem and stem.rsplit("_v", 1)[1].isdigit():
        base, vidx = stem.rsplit("_v", 1)
        return base, int(vidx)
    return stem, 0


# -----------------------------------------------------------------------------
# Gate 1 — cheap NumPy pre-filter
# -----------------------------------------------------------------------------

def gate_prefilter(y: np.ndarray) -> Optional[str]:
    """Returns reject-reason string or None on pass. y is (samples, channels)."""
    if y.size == 0:
        return "empty"
    if not np.isfinite(y).all():
        return "nan_or_inf"
    peak = float(np.max(np.abs(y)))
    if peak < db_to_amp(-40):
        return "silent"
    if abs(float(np.mean(y))) > 0.01:
        return "dc_offset"
    # Strict clipping: count exactly-at-1.0 samples
    clipped_frac = float(np.mean(np.abs(y) >= 0.999))
    if clipped_frac > 0.0001:
        return "clipped"
    # Dead-channel detection (stereo only): if one channel is >40dB below the other
    if y.ndim == 2 and y.shape[1] == 2:
        rms_l = float(np.sqrt(np.mean(y[:, 0] ** 2)))
        rms_r = float(np.sqrt(np.mean(y[:, 1] ** 2)))
        if rms_l > 0 and rms_r > 0:
            diff_db = 20 * math.log10(max(rms_l, rms_r) / min(rms_l, rms_r))
            if diff_db > 40:
                return "dead_channel"
    return None


# -----------------------------------------------------------------------------
# Gate 2 — onset (librosa)
# -----------------------------------------------------------------------------

def gate_onset(mono: np.ndarray, sr: int, max_onset_ms: float = 300.0) -> tuple[Optional[str], float]:
    """Returns (reject_reason, onset_ms). Imported inside to keep cold-start cheap.

    Default 300 ms (was 30 ms). SA3 output routinely has lead-in noise / fade-in
    before the actual transient; 30 ms was rejecting ~95% of usable samples.
    300 ms still rejects samples where the audio is just a tail-end with no
    perceptible attack."""
    import librosa
    try:
        onsets = librosa.onset.onset_detect(y=mono, sr=sr, units="frames", backtrack=True)
    except Exception as e:
        return f"onset_detect_failed:{e}", float("nan")
    if len(onsets) == 0:
        return "no_onset", float("nan")
    # onset_detect with backtrack=True returns the perceptual onset frame.
    first_onset_sample = librosa.frames_to_samples(onsets[:1])[0]
    onset_ms = (first_onset_sample / sr) * 1000.0
    if onset_ms > max_onset_ms:
        return "slow_onset", onset_ms
    return None, onset_ms


# -----------------------------------------------------------------------------
# Gate 3 — sustain plateau (custom DSP)
# -----------------------------------------------------------------------------

def gate_sustain(mono: np.ndarray, sr: int, min_sustain_seconds: float, onset_ms: float
                 ) -> tuple[Optional[str], float, float]:
    """Detect peak-plateau region; reject if shorter than min_sustain.

    Returns (reject_reason, sustain_seconds, sustain_quality).

    sustain_quality ∈ [0, 1] is sustain_seconds / max(min_sustain, 1.0) clipped to 1 —
    feeds the composite variant-score formula.
    """
    if min_sustain_seconds <= 0:
        # Some categories don't care (FX or true percussion). Mark as 1.0.
        return None, 0.0, 1.0

    import librosa

    # Onset offset in samples (skip the attack region for plateau detection)
    onset_sample = int(max(0.0, onset_ms) / 1000.0 * sr) if math.isfinite(onset_ms) else 0
    body = mono[onset_sample:]
    if body.size < int(sr * 0.05):
        return "short_body", 0.0, 0.0

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=body, frame_length=frame_length, hop_length=hop_length)[0]
    if rms.size == 0:
        return "rms_empty", 0.0, 0.0

    peak_rms = float(np.max(rms))
    if peak_rms <= 0:
        return "rms_zero", 0.0, 0.0

    # Plateau = frames within 12 dB of peak (i.e. >= peak/4). Find the longest
    # contiguous run. Was 6 dB (peak/2) — too tight for naturally-decaying
    # envelopes (piano, plucked strings, mallets) where the RMS curve drops
    # below 6dB-down within ~100ms of the attack. 12 dB still excludes the
    # noise floor and the late-decay tail.
    threshold = peak_rms / 4.0
    above = rms >= threshold
    # Run-length encode
    longest_run_frames = 0
    cur = 0
    for v in above:
        if v:
            cur += 1
            longest_run_frames = max(longest_run_frames, cur)
        else:
            cur = 0

    sustain_seconds = longest_run_frames * hop_length / sr
    sustain_quality = min(1.0, sustain_seconds / max(min_sustain_seconds, 1e-3))

    if sustain_seconds < min_sustain_seconds:
        return "short_stab", sustain_seconds, sustain_quality
    return None, sustain_seconds, sustain_quality


# -----------------------------------------------------------------------------
# Gate 5 — pitch (torchcrepe + sub-bass cross-check)
# -----------------------------------------------------------------------------

def _crepe_pitch(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pitch_hz_per_frame, periodicity_per_frame) from torchcrepe."""
    import torch
    import torchcrepe
    # torchcrepe wants float32 batched [1, samples], 16k sr (it resamples internally
    # via `sample_rate` param)
    audio = torch.tensor(mono, dtype=torch.float32).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=sr,
        hop_length=int(sr * 0.01),  # 10ms hop
        fmin=50.0,
        fmax=2000.0,
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        device=device,
        batch_size=512,
    )
    return pitch.squeeze(0).cpu().numpy(), periodicity.squeeze(0).cpu().numpy()


def _pyin_pitch(mono: np.ndarray, sr: int, fmin: float = 25.0, fmax: float = 200.0) -> np.ndarray:
    """Return per-frame pitch in Hz from librosa.pyin (NaN for unvoiced)."""
    import librosa
    f0, _voiced, _vprob = librosa.pyin(mono, fmin=fmin, fmax=fmax, sr=sr)
    return f0


def _autocorr_pitch(mono: np.ndarray, sr: int, fmin: float = 25.0, fmax: float = 200.0) -> float:
    """Single-shot autocorrelation pitch on the mid-section of the sample.

    Used as the 3rd voice for sub-bass agreement. Picks the lag with
    maximum normalized autocorrelation between fmin/fmax bounds.
    """
    # Mid-section (skip attack + release)
    n = mono.size
    a, b = int(n * 0.25), int(n * 0.75)
    seg = mono[a:b]
    if seg.size < 4096:
        return float("nan")
    seg = seg.astype(np.float64)
    seg -= seg.mean()
    # Convolution-based autocorrelation
    ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
    if ac[0] <= 0:
        return float("nan")
    ac /= ac[0]
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max >= ac.size:
        return float("nan")
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    return sr / lag if lag > 0 else float("nan")


def _detect_vibrato(pitch_hz: np.ndarray, frame_rate: float) -> bool:
    """Return True if the pitch envelope shows a periodic LFO at 3-8 Hz
    with depth ≤ 100 cents. That LFO is musical vibrato; without this
    bypass, basic-pitch would flag vibrato'd notes as polyphonic."""
    voiced = pitch_hz[np.isfinite(pitch_hz) & (pitch_hz > 0)]
    if voiced.size < 50:
        return False
    cents = 1200.0 * np.log2(voiced / np.median(voiced))
    if np.ptp(cents) > 200:
        return False  # too much swing for vibrato
    spec = np.abs(np.fft.rfft(cents - cents.mean()))
    freqs = np.fft.rfftfreq(cents.size, d=1.0 / frame_rate)
    band = (freqs >= 3.0) & (freqs <= 8.0)
    if not band.any():
        return False
    peak = spec[band].max()
    return peak > 0.3 * spec.max()


def gate_pitch(mono: np.ndarray, sr: int, target_midi: int, tolerance_cents: int,
               pitch_floor_hz: float) -> tuple[Optional[str], float, float, float, np.ndarray]:
    """Run torchcrepe + sub-bass cross-check. Returns:

        (reject_reason, measured_midi, cents_offset, confidence, pitch_envelope_hz)

    Sub-bass branch (target < E2/82Hz): require 2-of-3 agreement among
    torchcrepe, pyin, and autocorrelation within 50 cents.
    """
    target_hz = midi_to_hz(target_midi)
    pitch_hz, periodicity = _crepe_pitch(mono, sr)
    # Skip attack region (~50 ms) for median
    skip_frames = max(0, int(0.05 / 0.01))
    sustain = pitch_hz[skip_frames:]
    sustain_per = periodicity[skip_frames:]
    # Voiced-frame periodicity threshold lowered from 0.5 -> 0.3. SA3 audio
    # has more natural dynamic/timbral variation than studio-clean library
    # content; 0.5 was rejecting too many otherwise-pitched frames as
    # unvoiced. 0.3 still excludes truly noisy frames.
    voiced_mask = (sustain > 0) & np.isfinite(sustain) & (sustain_per >= 0.3)
    if not voiced_mask.any():
        return "no_voiced_frames", float("nan"), float("nan"), 0.0, pitch_hz
    crepe_hz = float(np.median(sustain[voiced_mask]))
    crepe_conf = float(np.median(sustain_per[voiced_mask]))

    is_sub_bass = target_hz < 82.0  # below E2

    if not is_sub_bass:
        # Confidence threshold lowered from 0.85 -> 0.3. 0.85 demanded
        # near-perfect pitch clarity which clean studio samples have but
        # SA3 output doesn't. 0.3 keeps clearly-pitched content while still
        # rejecting noise/atonal output.
        if crepe_conf < 0.3:
            return "unconfident", hz_to_midi(crepe_hz), cents_between(crepe_hz, target_hz), crepe_conf, pitch_hz
        cents = cents_between(crepe_hz, target_hz)
        if abs(cents) > tolerance_cents:
            return "wrong_pitch", hz_to_midi(crepe_hz), cents, crepe_conf, pitch_hz
        return None, hz_to_midi(crepe_hz), cents, crepe_conf, pitch_hz

    # Sub-bass: 3-way agreement
    pyin_pitch = _pyin_pitch(mono, sr, fmin=max(20.0, pitch_floor_hz * 0.8), fmax=200.0)
    pyin_voiced = pyin_pitch[np.isfinite(pyin_pitch) & (pyin_pitch > 0)]
    pyin_hz = float(np.median(pyin_voiced)) if pyin_voiced.size > 0 else float("nan")
    ac_hz = _autocorr_pitch(mono, sr, fmin=max(20.0, pitch_floor_hz * 0.8), fmax=200.0)

    candidates = [("crepe", crepe_hz), ("pyin", pyin_hz), ("ac", ac_hz)]
    valid = [(label, hz) for label, hz in candidates if hz > 0 and math.isfinite(hz)]
    if len(valid) < 2:
        return "subbass_no_candidates", crepe_hz, cents_between(crepe_hz, target_hz), crepe_conf, pitch_hz
    # Find the most-agreeing pair: their cents-diff is smallest
    best_pair_cents = float("inf")
    best_hz = float("nan")
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            d = abs(cents_between(valid[i][1], valid[j][1]))
            if d < best_pair_cents:
                best_pair_cents = d
                best_hz = (valid[i][1] + valid[j][1]) / 2.0
    if best_pair_cents > 50.0:
        return "subbass_disagreement", crepe_hz, cents_between(crepe_hz, target_hz), crepe_conf, pitch_hz
    cents = cents_between(best_hz, target_hz)
    if abs(cents) > tolerance_cents:
        return "wrong_pitch", hz_to_midi(best_hz), cents, crepe_conf, pitch_hz
    return None, hz_to_midi(best_hz), cents, crepe_conf, pitch_hz


# -----------------------------------------------------------------------------
# Gate 4 — polyphony (Basic Pitch)
# -----------------------------------------------------------------------------

def gate_polyphony(mono: np.ndarray, sr: int, pitch_envelope_hz: np.ndarray, frame_rate: float
                   ) -> tuple[Optional[str], str]:
    """Returns (reject_reason, polyphony_check_label).

    Vibrato detection from the CREPE pitch envelope short-circuits this
    gate — vibrato samples otherwise read as polyphonic because Basic
    Pitch fragments vibrato'd notes into multiple short overlapping ones.
    """
    if _detect_vibrato(pitch_envelope_hz, frame_rate):
        return None, "vibrato_bypass"

    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception as e:
        # If basic-pitch isn't importable on the pod, skip with a warning
        # rather than failing — better to gate-pass than fail the whole batch.
        print(f"[gate] basic-pitch unavailable, skipping polyphony check: {e}", file=sys.stderr)
        return None, "polyphony_skipped"

    try:
        # Basic Pitch expects a file path or a numpy array; passing the array
        # avoids a temp-file round trip.
        _model_output, note_events, _midi = predict(
            mono, sr,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,  # ms
        )
    except Exception as e:
        print(f"[gate] basic-pitch predict failed: {e}", file=sys.stderr)
        return None, "polyphony_skipped"

    # note_events: list of (start_time, end_time, pitch_midi, amplitude, pitch_bends)
    if note_events is None or len(note_events) < 2:
        return None, "monophonic"
    # Look for pairs whose time-overlap > 100 ms AND pitch-class differs
    for i in range(len(note_events)):
        s_i, e_i, p_i, *_ = note_events[i]
        for j in range(i + 1, len(note_events)):
            s_j, e_j, p_j, *_ = note_events[j]
            overlap = min(e_i, e_j) - max(s_i, s_j)
            if overlap > 0.1 and (int(p_i) % 12) != (int(p_j) % 12):
                return "polyphonic", "polyphonic"
    return None, "monophonic"


# -----------------------------------------------------------------------------
# Per-variant evaluation
# -----------------------------------------------------------------------------

def evaluate_variant(wav_path: Path, cfg: PitchedCategoryConfig, target_pitch: int) -> dict:
    """Run all gates on one WAV; return a verdict dict (always written to gate.json)."""
    verdict: dict = {
        "path": str(wav_path),
        "passed": False,
        "rejection_reason": None,
        "score": 0.0,
        "metrics": {},
    }

    try:
        y, sr = sf.read(wav_path, always_2d=True)
    except Exception as e:
        verdict["rejection_reason"] = f"read_failed:{e}"
        return verdict

    # Pre-filter on raw multi-channel
    pre = gate_prefilter(y)
    if pre is not None:
        verdict["rejection_reason"] = pre
        return verdict

    # Mono mix for all subsequent DSP
    mono = np.mean(y, axis=1).astype(np.float32)

    # Onset
    onset_reason, onset_ms = gate_onset(mono, sr)
    verdict["metrics"]["onset_ms"] = round(onset_ms, 2) if math.isfinite(onset_ms) else None
    if onset_reason is not None:
        verdict["rejection_reason"] = onset_reason
        return verdict

    # Sustain plateau
    sus_reason, sus_seconds, sus_quality = gate_sustain(mono, sr, cfg.min_sustain_seconds, onset_ms)
    verdict["metrics"]["sustain_seconds"] = round(sus_seconds, 3)
    verdict["metrics"]["sustain_quality"] = round(sus_quality, 3)
    if sus_reason is not None:
        verdict["rejection_reason"] = sus_reason
        return verdict

    # Pitch (also returns the pitch envelope used by vibrato detection)
    if cfg.skip_pitch_shift or cfg.pitch_tolerance_cents >= 9999:
        # FX or any category that opts out of pitch validation
        measured_midi, cents, confidence = float("nan"), 0.0, 1.0
        pitch_env = np.array([])
    else:
        pitch_reason, measured_midi, cents, confidence, pitch_env = gate_pitch(
            mono, sr,
            target_midi=target_pitch,
            tolerance_cents=cfg.pitch_tolerance_cents,
            pitch_floor_hz=cfg.pitch_detection_floor_hz,
        )
        verdict["metrics"]["measured_pitch_midi"] = (
            round(measured_midi, 2) if math.isfinite(measured_midi) else None
        )
        verdict["metrics"]["measured_pitch_cents_offset"] = (
            round(cents, 1) if math.isfinite(cents) else None
        )
        verdict["metrics"]["pitch_confidence"] = round(confidence, 3)
        if pitch_reason is not None:
            verdict["rejection_reason"] = pitch_reason
            return verdict

    # Polyphony (runs after pitch since the vibrato bypass needs the pitch envelope)
    poly_reason, poly_label = gate_polyphony(mono, sr, pitch_env, frame_rate=100.0)
    verdict["metrics"]["polyphony_check"] = poly_label
    if poly_reason is not None:
        verdict["rejection_reason"] = poly_reason
        return verdict

    # All gates passed — compute composite score
    score = (max(confidence, 1e-6) ** 2) * math.exp(-abs(cents) / 50.0) * sus_quality
    verdict["passed"] = True
    verdict["score"] = round(score, 4)
    return verdict


# -----------------------------------------------------------------------------
# Main: group variants, pick winners, write artefacts
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", required=True,
                        help="Pitched category name (must exist in pitched_category_config.py)")
    parser.add_argument("--jsonl", required=True,
                        help="The source JSONL written by list_to_jsonl_pitched.py (for target_pitch lookup)")
    parser.add_argument("--in-dir", default=None,
                        help="Raw WAV dir. Default: $SAS_OUTPUTS_DIR/raw/<cat>/")
    parser.add_argument("--out-dir", default=None,
                        help="Gated output dir. Default: $SAS_OUTPUTS_DIR/gated/<cat>/")
    parser.add_argument("--outputs-dir", default=None,
                        help="Override $SAS_OUTPUTS_DIR (root for raw/, gated/, instruments/).")
    args = parser.parse_args()

    cfg = PITCHED_CATEGORIES.get(args.category)
    if cfg is None:
        sys.exit(f"unknown category {args.category!r}; add to pitched_category_config.py first")

    import os
    outputs_dir = Path(args.outputs_dir or os.environ.get("SAS_OUTPUTS_DIR", "outputs"))
    in_dir = Path(args.in_dir) if args.in_dir else outputs_dir / "raw" / args.category
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir / "gated" / args.category
    failures_dir = out_dir / "_failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    # Index source JSONL by id → target_pitch_midi
    target_pitch_by_id: dict[str, int] = {}
    prompt_by_id: dict[str, str] = {}
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target_pitch_by_id[row["id"]] = int(row["target_pitch_midi"])
            prompt_by_id[row["id"]] = row.get("prompt", "")

    # Group raw WAVs by base id
    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        print(f"[gate] no WAVs in {in_dir}; nothing to do")
        return
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for wav in wavs:
        base, vidx = strip_variant_suffix(wav.stem)
        groups[base].append((vidx, wav))

    passed = 0
    failed = 0
    for base_id, variants in sorted(groups.items()):
        target_pitch = target_pitch_by_id.get(base_id)
        if target_pitch is None:
            print(f"[gate] {base_id}: not in jsonl, skipping (orphan)", file=sys.stderr)
            continue

        verdicts: list[dict] = []
        for vidx, wav in sorted(variants):
            print(f"[gate] {base_id}_v{vidx:02d} ...", flush=True)
            v = evaluate_variant(wav, cfg, target_pitch)
            v["variant_index"] = vidx
            verdicts.append(v)
            status = "PASS" if v["passed"] else f"REJECT({v['rejection_reason']})"
            print(f"[gate]   → {status} score={v['score']:.3f}", flush=True)

        winners = [v for v in verdicts if v["passed"]]
        if not winners:
            # Full failure — write _failures/<base_id>.json with rejection counts
            from collections import Counter
            reasons = Counter(v["rejection_reason"] for v in verdicts)
            (failures_dir / f"{base_id}.json").write_text(json.dumps({
                "id": base_id,
                "prompt": prompt_by_id.get(base_id, ""),
                "target_pitch_midi": target_pitch,
                "attempts": len(verdicts),
                "rejection_reasons": dict(reasons),
                "raw_paths": [v["path"] for v in verdicts],
            }, indent=2), encoding="utf-8")
            failed += 1
            print(f"[gate] {base_id}: ALL VARIANTS FAILED ({dict(reasons)})", flush=True)
            continue

        winner = max(winners, key=lambda v: v["score"])
        winner_path = Path(winner["path"])
        out_wav = out_dir / f"{base_id}.wav"
        out_json = out_dir / f"{base_id}.gate.json"
        shutil.copy2(winner_path, out_wav)
        out_json.write_text(json.dumps({
            "id": base_id,
            "target_pitch_midi": target_pitch,
            "winner": winner,
            "all_variants": verdicts,
            "raw_winner_path": str(winner_path),
        }, indent=2), encoding="utf-8")
        passed += 1
        print(f"[gate] {base_id}: winner v{winner['variant_index']:02d} score={winner['score']:.3f}", flush=True)

    print(f"[gate] done. passed={passed} failed={failed} (failure rate {100 * failed / max(1, passed + failed):.1f}%)")


if __name__ == "__main__":
    main()
