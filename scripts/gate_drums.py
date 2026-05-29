"""Quality-gate + best-of-N selection for the drum / one-shot pipeline.

Drums are unpitched, so (unlike gate_pitched) there's no pitch check. Instead,
per the category PROFILE (drum_gate_config.py), each SA3 candidate runs through:

  1. Prefilter  — NaN/Inf, silent, DC offset, clipping, dead channel (NumPy).
  2. Attack     — a crisp transient must land within max_onset_ms (percussive /
                  cymbal / sub). Skipped for fx (risers/sweeps build up).
  3. Decay      — the tail must fall well below the peak (reject sustained
                  drones / pads where a one-shot was asked for). Skipped for fx.
  4. Single-hit — reject multiple spaced transients (a loop / roll leaked in).
                  Only for single_hit categories.
  5. Spectral   — lenient hard reject when grossly off the category's centroid
                  band (e.g. an all-treble "kick"); otherwise a soft score.

Variants that pass are scored by a composite (centroid match × decay clean ×
attack × punch) and the BEST per (prompt) is copied to:

    outputs/gated_drums/<cat>/<id>.wav
    outputs/gated_drums/<cat>/<id>.drumgate.json
    outputs/gated_drums/<cat>/_metadata/<id>.json   (winner's raw sidecar, for postprocess)

postprocess_oneshots.py then runs on gated_drums/ (trim / LUFS / mono / tags).
Groups where every variant fails land in _failures/<id>.json.

Pure NumPy + librosa (lazy import) — no torch/GPU; runs on the pod or locally.

Usage:
  python scripts/gate_drums.py --category kick
  python scripts/gate_drums.py --category kick --in-dir outputs/raw/kick --out-dir outputs/gated_drums/kick
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drum_gate_config import get_profile, DrumProfile

_VAR_RE = re.compile(r"_v(\d{2})$")


def strip_variant_suffix(stem: str) -> tuple[str, int]:
    m = _VAR_RE.search(stem)
    if m:
        return stem[: m.start()], int(m.group(1))
    return stem, -1


# -----------------------------------------------------------------------------
# DSP helpers (pure NumPy where possible; librosa lazy-imported)
# -----------------------------------------------------------------------------

def gate_prefilter(y: np.ndarray) -> Optional[str]:
    """Cheap rejects on the raw (samples, channels) buffer."""
    if y.size == 0:
        return "empty"
    if not np.all(np.isfinite(y)):
        return "nan_or_inf"
    mono = np.mean(y, axis=1) if y.ndim == 2 else y
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-4:
        return "silent"
    if abs(float(np.mean(mono))) > 0.1 * peak:
        return "dc_offset"
    if float(np.mean(np.abs(mono) >= 0.999)) > 0.01:
        return "clipped"
    if y.ndim == 2 and y.shape[1] == 2:
        pk = np.max(np.abs(y), axis=0)
        if float(np.min(pk)) < 1e-4 < float(np.max(pk)):
            return "dead_channel"
    return None


def _rms_envelope(mono: np.ndarray, sr: int, hop: int = 256, win: int = 1024):
    import librosa
    rms = librosa.feature.rms(y=mono, frame_length=win, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return rms, times


def attack_ms(mono: np.ndarray, sr: int) -> float:
    """Time from first-audible sample to the peak (ms) — a crisp one-shot rises
    fast. Pure NumPy."""
    peak = float(np.max(np.abs(mono)))
    if peak <= 0:
        return float("inf")
    above = np.where(np.abs(mono) >= 0.1 * peak)[0]
    if above.size == 0:
        return float("inf")
    onset_idx = int(above[0])
    peak_idx = int(np.argmax(np.abs(mono)))
    return max(0.0, (peak_idx - onset_idx) / sr * 1000.0)


def decay_ratio(mono: np.ndarray, sr: int) -> float:
    """RMS of the last 15% / peak RMS. Near 0 = decays cleanly; near 1 = a
    sustained drone that never falls off."""
    rms, _ = _rms_envelope(mono, sr)
    if rms.size < 4:
        return 0.0
    peak_rms = float(np.max(rms))
    if peak_rms <= 0:
        return 1.0
    tail = rms[int(len(rms) * 0.85):]
    tail_rms = float(np.mean(tail)) if tail.size else 0.0
    return tail_rms / peak_rms


def count_reattacks(mono: np.ndarray, sr: int) -> int:
    """Count distinct re-attacks via the RMS envelope: after the main hit, every
    time the envelope falls near-silent (<=15% of peak) and then rises back to a
    strong level (>=50% of peak) is another hit — i.e. a loop/roll, not a single
    one-shot. A clean one-shot decays monotonically and returns 0. Robust to
    decaying tails / layered claps (which never drop near-silent between)."""
    rms, _ = _rms_envelope(mono, sr)
    if rms.size < 4:
        return 0
    peak = float(np.max(rms))
    if peak <= 0:
        return 0
    hi, lo = 0.5 * peak, 0.15 * peak
    count = 0
    seen_hit = False
    armed = False
    for v in rms:
        if v >= hi:
            if armed:
                count += 1
                armed = False
            seen_hit = True
        elif v <= lo and seen_hit:
            armed = True
    return count


def spectral_centroid_hz(mono: np.ndarray, sr: int) -> float:
    """Median spectral centroid over the ACTIVE region only — frames at >=15% of
    peak RMS (the actual hit), not the whole buffer.

    Measuring over the whole file is wrong for a short one-shot in a long file:
    the decay / noise-floor tail is broadband, so each tail frame's centroid sits
    near sr/4, and with ~80% of frames being tail the median tracks the noise
    floor rather than the sound. A faint -60 dBFS tail alone pushes a clean 60 Hz
    kick's whole-file median from ~60 Hz to ~11 kHz, which then trips `off_band`
    and nuked every low-frequency category (kick/toms) while high-band categories
    (shaker/cymbals/hats) coincidentally still matched. Restricting to loud frames
    makes the centroid reflect the hit, not the silence after it."""
    import librosa
    hop = 512
    try:
        sc = librosa.feature.spectral_centroid(y=mono, sr=sr, hop_length=hop)[0]
        rms = librosa.feature.rms(y=mono, hop_length=hop)[0]
    except Exception:
        return float("nan")
    n = min(len(sc), len(rms))
    sc, rms = sc[:n], rms[:n]
    valid = np.isfinite(sc) & (sc > 0)
    if not valid.any():
        return float("nan")
    peak = float(np.max(rms))
    active = valid & (rms >= 0.15 * peak) if peak > 0 else valid
    sel = sc[active] if np.any(active) else sc[valid]
    return float(np.median(sel))


def crest_factor(mono: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    return peak / rms if rms > 0 else 0.0


# -----------------------------------------------------------------------------
# Evaluate one variant
# -----------------------------------------------------------------------------

def evaluate_variant(wav_path: Path, prof: DrumProfile) -> dict:
    verdict: dict = {"path": str(wav_path), "passed": False, "rejection_reason": None,
                     "score": 0.0, "metrics": {}}
    try:
        y, sr = sf.read(str(wav_path), always_2d=True)
    except Exception as e:
        verdict["rejection_reason"] = f"read_failed:{e}"
        return verdict

    pre = gate_prefilter(y)
    if pre is not None:
        verdict["rejection_reason"] = pre
        return verdict

    mono = np.mean(y, axis=1).astype(np.float32)
    dur = len(mono) / sr
    verdict["metrics"]["duration_s"] = round(dur, 3)
    if dur < prof.min_duration_s:
        verdict["rejection_reason"] = "too_short"
        return verdict

    is_fx = prof.kind == "fx"

    # Attack (percussive/cymbal/sub)
    atk = attack_ms(mono, sr)
    verdict["metrics"]["attack_ms"] = round(atk, 1) if math.isfinite(atk) else None
    if not is_fx and prof.max_onset_ms > 0 and atk > prof.max_onset_ms:
        verdict["rejection_reason"] = "slow_attack"
        return verdict

    # Decay (reject sustained drones where a one-shot was asked for)
    dr = decay_ratio(mono, sr)
    verdict["metrics"]["decay_ratio"] = round(dr, 3)
    if prof.expect_decay and dr > 0.6:
        verdict["rejection_reason"] = "sustained_no_decay"
        return verdict

    # Single-hit (loop/roll rejection)
    if prof.single_hit:
        n = count_reattacks(mono, sr)
        verdict["metrics"]["reattacks"] = n
        if n >= 1:
            verdict["rejection_reason"] = "multi_hit_loop"
            return verdict

    # Spectral centroid — lenient hard reject for gross mismatch, else soft score
    sc = spectral_centroid_hz(mono, sr)
    verdict["metrics"]["centroid_hz"] = round(sc, 1) if math.isfinite(sc) else None
    lo, hi = prof.centroid_hz
    if math.isfinite(sc) and not is_fx and (sc < lo / 3.0 or sc > hi * 3.0):
        verdict["rejection_reason"] = "off_band"
        return verdict

    # ---- passed: composite score (higher = better candidate) ----
    center = math.sqrt(lo * hi)
    if math.isfinite(sc) and sc > 0:
        centroid_match = math.exp(-((math.log(sc) - math.log(center)) / 0.9) ** 2)
    else:
        centroid_match = 0.5
    decay_clean = 1.0 - min(1.0, dr) if prof.expect_decay else 1.0
    if is_fx or prof.max_onset_ms <= 0:
        attack_score = 1.0
    else:
        attack_score = math.exp(-atk / max(prof.max_onset_ms, 1.0))
    crest = crest_factor(mono)
    punch = min(1.0, crest / 12.0)  # crest ~12 dB-ish is a punchy transient

    score = centroid_match * (0.4 + 0.6 * decay_clean) * (0.4 + 0.6 * attack_score) * (0.5 + 0.5 * punch)
    verdict["metrics"]["crest_factor"] = round(crest, 2)
    verdict["passed"] = True
    verdict["score"] = round(score, 4)
    return verdict


# -----------------------------------------------------------------------------
# Main: group variants, pick winners, copy + sidecar
# -----------------------------------------------------------------------------

def run_gate(category: str, in_dir: Path, out_dir: Path,
             only_ids: Optional[set] = None) -> None:
    prof = get_profile(category)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_out = out_dir / "_metadata"
    meta_out.mkdir(parents=True, exist_ok=True)
    failures_dir = out_dir / "_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    raw_meta_dir = in_dir / "_metadata"

    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        print(f"[gate_drums] no WAVs in {in_dir}; nothing to do")
        return

    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for wav in wavs:
        base, vidx = strip_variant_suffix(wav.stem)
        if only_ids is not None and base not in only_ids:
            continue  # retry round: only re-gate the ids that previously failed
        groups[base].append((vidx, wav))

    passed = 0
    failed = 0
    reasons: Counter = Counter()
    for base_id, variants in sorted(groups.items()):
        verdicts = []
        for vidx, wav in sorted(variants):
            v = evaluate_variant(wav, prof)
            v["variant_index"] = vidx
            verdicts.append(v)
            if not v["passed"]:
                reasons[v["rejection_reason"]] += 1

        winners = [v for v in verdicts if v["passed"]]
        if not winners:
            (failures_dir / f"{base_id}.json").write_text(json.dumps({
                "id": base_id,
                "category": category,
                "attempts": len(verdicts),
                "rejection_reasons": dict(Counter(v["rejection_reason"] for v in verdicts)),
            }, indent=2), encoding="utf-8")
            failed += 1
            continue

        winner = max(winners, key=lambda v: v["score"])
        winner_path = Path(winner["path"])
        vidx = winner["variant_index"]
        # Clear any stale failure marker from an earlier round (now it passes).
        stale = failures_dir / f"{base_id}.json"
        if stale.exists():
            stale.unlink()
        # Copy winner WAV (drop the _vNN suffix so postprocess/pack see one per prompt).
        shutil.copy2(winner_path, out_dir / f"{base_id}.wav")
        # Carry the winner's raw metadata sidecar across (postprocess reads the prompt from it).
        suffix = f"_v{vidx:02d}" if vidx >= 0 else ""
        src_meta = raw_meta_dir / f"{base_id}{suffix}.json"
        if src_meta.exists():
            shutil.copy2(src_meta, meta_out / f"{base_id}.json")
        (out_dir / f"{base_id}.drumgate.json").write_text(json.dumps({
            "id": base_id, "category": category, "winner": winner,
            "all_variants": verdicts, "raw_winner_path": str(winner_path),
        }, indent=2), encoding="utf-8")
        passed += 1

    print(f"[gate_drums] {category}: kept {passed} / {len(groups)} prompts "
          f"({failed} all-fail). reject tally: {dict(reasons.most_common())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", required=True)
    parser.add_argument("--in-dir", default=None, help="Raw dir. Default $SAS_OUTPUTS_DIR/raw/<cat>/")
    parser.add_argument("--out-dir", default=None,
                        help="Gated dir. Default $SAS_OUTPUTS_DIR/gated_drums/<cat>/")
    parser.add_argument("--outputs-dir", default=None, help="Override $SAS_OUTPUTS_DIR")
    parser.add_argument("--only-ids", default=None,
                        help="File with one base id per line; restrict the gate to these "
                             "(retry rounds — re-gate only the prompts that previously failed).")
    args = parser.parse_args()

    import os
    outputs_dir = Path(args.outputs_dir or os.environ.get("SAS_OUTPUTS_DIR", "outputs"))
    in_dir = Path(args.in_dir) if args.in_dir else outputs_dir / "raw" / args.category
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir / "gated_drums" / args.category
    only_ids = None
    if args.only_ids and Path(args.only_ids).exists():
        only_ids = {ln.strip() for ln in Path(args.only_ids).read_text().splitlines() if ln.strip()}
    run_gate(args.category, in_dir, out_dir, only_ids=only_ids)


if __name__ == "__main__":
    main()
