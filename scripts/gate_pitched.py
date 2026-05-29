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


# -----------------------------------------------------------------------------
# Spectral fundamental (monophonic one-shots) — the robust pitch NUMBER
# -----------------------------------------------------------------------------
#
# torchcrepe is excellent at VOICING ("is this a clearly pitched note?") but its
# per-frame pitch *median* is fragile on harmonically-rich sustains: the track
# straddles the fundamental and its partials, and `np.median` settles on a
# phantom value BETWEEN them. A C3 (130.8 Hz) synth whose frames split across
# C3 / G3 / C4 medians to ~170 Hz (≈F3) — a perfect 4th sharp — which then
# mislabels every sampler zone. (This shipped the entire synth library a 4th
# out of tune; see tests/test_pitch_detection.py for the reproduction.)
#
# These samples are MONOPHONIC single notes, so a harmonic-sum over the averaged
# magnitude spectrum is both simpler and far more reliable. Validated on the
# full instrument library: pitch-class accuracy 15% -> 70%.

def _avg_magnitude_spectrum(mono: np.ndarray, sr: int, fft_size: int = 1 << 15,
                            max_windows: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum averaged over several sustain windows (skips ~100 ms
    of attack). Averaging stabilises the harmonic structure vs a single frame."""
    y = np.asarray(mono, dtype=np.float64)
    y = y[int(0.1 * sr):]
    if y.size < fft_size:
        y = np.pad(y, (0, fft_size - y.size))
    window = np.hanning(fft_size)
    hop = fft_size // 2
    mags = []
    for start in range(0, y.size - fft_size + 1, hop):
        mags.append(np.abs(np.fft.rfft(y[start:start + fft_size] * window)))
        if len(mags) >= max_windows:
            break
    if not mags:
        mags = [np.abs(np.fft.rfft(y[:fft_size] * window))]
    return np.mean(mags, axis=0), np.fft.rfftfreq(fft_size, 1.0 / sr)


def _spectral_peaks(mag: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float,
                    top_n: int = 30) -> list[tuple[float, float]]:
    """Strongest spectral peaks in [fmin, fmax], de-duplicated to ~6 Hz, returned
    strongest-first. Restricting fundamental candidates to *real* peaks avoids
    the subharmonic phantoms that plain HPS produces."""
    band = (freqs >= fmin) & (freqs <= fmax)
    fb, mb = freqs[band], mag[band]
    out: list[tuple[float, float]] = []
    for i in np.argsort(mb)[::-1]:
        f = float(fb[i])
        if any(abs(f - pf) < 6.0 for pf, _ in out):
            continue
        out.append((f, float(mb[i])))
        if len(out) >= top_n:
            break
    return out


def _harmonic_sum_score(mag: np.ndarray, freqs: np.ndarray, f0: float,
                        n_harmonics: int = 6) -> float:
    """Sum of magnitude at f0, 2·f0 … n·f0 with 1/h weighting. The 1/h weight
    rewards energy at the fundamental + low harmonics, so a true fundamental
    outscores its own subharmonic (whose energy sits in *upper* harmonics)."""
    df = freqs[1] - freqs[0]
    score = 0.0
    for h in range(1, n_harmonics + 1):
        fh = f0 * h
        if fh > freqs[-1]:
            break
        b = int(round(fh / df))
        win = mag[max(0, b - 2):b + 3]
        score += (float(win.max()) if win.size else 0.0) / h
    return score


def _refine_peak(mag: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    """Snap f0 to the local magnitude maximum within ±6% (sub-bin precision so
    the cents offset is accurate)."""
    band = (freqs >= f0 * 0.94) & (freqs <= f0 * 1.06)
    if not band.any():
        return f0
    fb = freqs[band]
    return float(fb[np.argmax(mag[band])])


def spectral_fundamental_midi(mono: np.ndarray, sr: int, target_midi: int) -> float:
    """Robust fundamental (in MIDI) for a monophonic instrument one-shot.

    Picks the spectral peak whose harmonic series best explains the spectrum
    (harmonic-sum, 1/h weighted). Octave guard: keep the detected octave when it
    is within ~1 octave of the requested target; otherwise re-seat the detected
    pitch CLASS in the requested register. That guards gross octave-ID blunders
    (e.g. reporting F6 for a C3 request) while still honouring the pipeline's
    "use the pitch SA3 actually produced" philosophy in the common case.
    Returns NaN when no clear peak exists.
    """
    mag, freqs = _avg_magnitude_spectrum(mono, sr)
    fmin = max(30.0, midi_to_hz(target_midi - 15))
    fmax = min(float(freqs[-1]), midi_to_hz(target_midi + 24))
    candidates = _spectral_peaks(mag, freqs, fmin, fmax)
    # candidates are strongest-first; a ~zero top peak means silence / no tone.
    if not candidates or candidates[0][1] <= 1e-9:
        return float("nan")
    best_f0 = max(candidates, key=lambda c: _harmonic_sum_score(mag, freqs, c[0]))[0]
    measured = hz_to_midi(_refine_peak(mag, freqs, best_f0))
    # Octave placement: keep the detected pitch CLASS (what spectral detection
    # reliably nails) but seat it in the octave nearest the requested target —
    # the musically-intended register, and robust to the fundamental-vs-loudest
    # -partial octave ambiguity (a C2 tone whose loudest partial is C3, etc.).
    # An octave mislabel is "right note, wrong register"; a pitch-CLASS mislabel
    # is "out of key" — so we optimise class correctness, not absolute octave.
    # The sub-semitone offset is preserved so enrich's fine pitch-correction
    # still lands the sample exactly on a MIDI note.
    frac = measured - round(measured)
    pitch_class = int(round(measured)) % 12
    base = target_midi - (target_midi % 12) + pitch_class
    nearest = min((base - 12, base, base + 12), key=lambda c: abs(c - target_midi))
    return float(nearest) + frac


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


# Targets at or below this MIDI note get the autocorrelation cross-check.
# Sub-bass (≈ E2 and lower, plus tuned-low percussion like timpani F2) is where
# spectral harmonic-sum is most prone to locking onto a partial/subharmonic and
# reporting the wrong pitch CLASS; time-domain autocorrelation tracks the true
# period of a low monophonic tone more reliably there. Above this, trust spectral.
#
# NB: we deliberately do NOT use librosa.pyin here. pyin pulls in numba and can
# hard-SEGFAULT under some numpy/numba ABI combos (observed on numpy 2.0 / py3.9)
# — and a segfault is uncatchable, so it would take down the whole gate process.
# Autocorrelation is pure NumPy: no JIT, no crash, and testable on CPU.
SUB_BASS_ENSEMBLE_CEIL_MIDI = 43  # G2


def _autocorr_pitch_conf(mono: np.ndarray, sr: int, fmin: float,
                         fmax: float) -> tuple[float, float]:
    """Autocorrelation pitch on the steady mid-section. Returns (hz, strength)
    where strength is the normalized autocorrelation at the chosen lag (0..1) —
    a clear period scores near 1, broadband noise scores low. (nan, 0.0) when no
    usable lag exists. Pure NumPy."""
    n = mono.size
    a, b = int(n * 0.25), int(n * 0.75)
    seg = mono[a:b]
    if seg.size < 4096:
        return float("nan"), 0.0
    seg = seg.astype(np.float64)
    seg -= seg.mean()
    ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
    if ac[0] <= 0:
        return float("nan"), 0.0
    ac /= ac[0]
    lag_min = max(1, int(sr / fmax))
    lag_max = int(sr / fmin)
    if lag_max >= ac.size or lag_max <= lag_min:
        return float("nan"), 0.0
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    return (sr / lag if lag > 0 else float("nan")), float(ac[lag])


def ensemble_octave_check(mono: np.ndarray, sr: int, spectral_midi: float,
                          target_midi: int, pitch_floor_hz: float,
                          info_out: Optional[dict] = None) -> float:
    """Confirm/repair the spectral fundamental for sub-bass targets using a
    time-domain autocorrelation cross-check.

    The spectral detector already seats the pitch CLASS in the octave nearest
    the target; this guards the remaining failure mode for low tones where
    harmonic-sum locks onto a partial/subharmonic and the CLASS itself is wrong.
    Overrides spectral only when autocorrelation finds a CLEAR period
    (strength ≥ 0.5) that disagrees with the spectral reading by ≥ 1.5 semitones.
    The autocorr pitch is re-seated to the octave nearest the target (same
    philosophy as spectral_fundamental_midi), preserving its sub-semitone
    fraction so enrich's fine correction still lands on a MIDI note. Defensive:
    any failure keeps the spectral value unchanged.
    """
    info = info_out if info_out is not None else {}
    info["spectral_midi"] = round(spectral_midi, 2) if math.isfinite(spectral_midi) else None
    info["autocorr_midi"] = None
    info["autocorr_strength"] = None
    info["ensemble_applied"] = False
    try:
        # 30 Hz floor keeps the autocorr lag-range below the octave-below period
        # for all our targets (lowest fundamental E1 ≈ 41 Hz), which removes
        # autocorrelation's classic octave-down ambiguity.
        lo = max(30.0, pitch_floor_hz * 0.5)
        hi = midi_to_hz(target_midi + 12)
        ac_hz, ac_strength = _autocorr_pitch_conf(mono, sr, lo, hi)
        ac_midi = hz_to_midi(ac_hz)
        info["autocorr_midi"] = round(ac_midi, 2) if math.isfinite(ac_midi) else None
        info["autocorr_strength"] = round(ac_strength, 3)

        if not math.isfinite(ac_midi) or ac_strength < 0.5:
            return spectral_midi  # no clear period — trust spectral

        frac = ac_midi - round(ac_midi)
        pitch_class = int(round(ac_midi)) % 12
        base = target_midi - (target_midi % 12) + pitch_class
        nearest = min((base - 12, base, base + 12), key=lambda c: abs(c - target_midi))
        ac_seated = float(nearest) + frac

        if math.isfinite(spectral_midi) and abs(ac_seated - spectral_midi) >= 1.5:
            info["ensemble_applied"] = True
            return ac_seated
        return spectral_midi
    except Exception:
        return spectral_midi


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
               pitch_floor_hz: float,
               metrics_out: Optional[dict] = None) -> tuple[Optional[str], float, float, float, np.ndarray]:
    """Confidence/voicing from torchcrepe; pitch NUMBER from the spectral
    fundamental. Returns:

        (reject_reason, measured_midi, cents_offset, confidence, pitch_envelope_hz)

    torchcrepe periodicity still gates voicing/atonality (its strength) and
    feeds the downstream vibrato detector. The measured pitch, however, comes
    from `spectral_fundamental_midi`: crepe's per-frame *median* lands on a
    phantom value between partials for harmonically-rich tones — that is the bug
    that shipped the synth library a perfect-4th sharp.

    For SUB-BASS targets (≤ G2) the spectral reading is additionally cross-checked
    by `ensemble_octave_check` (time-domain autocorrelation), which repairs the
    rare case where harmonic-sum locks onto a partial/subharmonic and the pitch
    CLASS itself is wrong. The estimates are written to `metrics_out` (when
    supplied) under spectral_midi/autocorr_midi/autocorr_strength/ensemble_applied.
    """
    target_hz = midi_to_hz(target_midi)
    pitch_hz, periodicity = _crepe_pitch(mono, sr)
    # Skip attack region (~50 ms)
    skip_frames = max(0, int(0.05 / 0.01))
    sustain = pitch_hz[skip_frames:]
    sustain_per = periodicity[skip_frames:]
    # Voiced-frame periodicity threshold 0.3 (see git history for the 0.5->0.3
    # rationale): SA3 output is less clean than studio samples.
    voiced_mask = (sustain > 0) & np.isfinite(sustain) & (sustain_per >= 0.3)
    if not voiced_mask.any():
        return "no_voiced_frames", float("nan"), float("nan"), 0.0, pitch_hz
    crepe_conf = float(np.median(sustain_per[voiced_mask]))

    # Voicing/atonality gate stays with crepe — periodicity is what it is good
    # at. (Confidence threshold 0.3; see git history for the 0.85->0.3 move.)
    if crepe_conf < 0.3:
        crepe_hz = float(np.median(sustain[voiced_mask]))
        return ("unconfident", hz_to_midi(crepe_hz),
                cents_between(crepe_hz, target_hz), crepe_conf, pitch_hz)

    # Pitch NUMBER from the spectral fundamental (robust on monophonic one-shots).
    measured_midi = spectral_fundamental_midi(mono, sr, target_midi)
    if not math.isfinite(measured_midi):
        # No clear spectral peak — fall back to the crepe median.
        measured_midi = hz_to_midi(float(np.median(sustain[voiced_mask])))

    # Sub-bass octave/class cross-check (pyin + autocorr). Restricted to low
    # targets so the extra librosa.pyin pass barely moves gate wall-clock.
    if target_midi <= SUB_BASS_ENSEMBLE_CEIL_MIDI:
        ens_info: dict = {}
        measured_midi = ensemble_octave_check(
            mono, sr, measured_midi, target_midi, pitch_floor_hz, info_out=ens_info
        )
        if metrics_out is not None:
            metrics_out["pitch_ensemble"] = ens_info

    cents = cents_between(midi_to_hz(measured_midi), target_hz)
    if abs(cents) > tolerance_cents:
        return "wrong_pitch", measured_midi, cents, crepe_conf, pitch_hz
    return None, measured_midi, cents, crepe_conf, pitch_hz


# -----------------------------------------------------------------------------
# Gate 4 — polyphony (Basic Pitch)
# -----------------------------------------------------------------------------

# Module-level cache for basic-pitch availability. Probed lazily on first
# call and remembered for the rest of the run, so we don't re-import +
# re-emit "Coremltools is not installed" etc. on every variant.
_BASIC_PITCH_PROBED = False
_BASIC_PITCH_PREDICT = None  # set to callable if available, else stays None


def _probe_basic_pitch() -> None:
    """Try to import + call basic_pitch.inference.predict once. If it works,
    cache the callable. If it fails (TF/numpy ABI mismatch is the usual cause
    on RunPod), cache the failure so subsequent calls are silent no-ops."""
    global _BASIC_PITCH_PROBED, _BASIC_PITCH_PREDICT
    if _BASIC_PITCH_PROBED:
        return
    _BASIC_PITCH_PROBED = True
    try:
        # Silence basic-pitch's own runtime-detection warnings before import.
        import logging
        logging.getLogger("root").setLevel(logging.ERROR)
        from basic_pitch.inference import predict
        # Smoke-test on a tiny silent buffer; if predict explodes (e.g. TF/numpy
        # ABI mismatch), we want to catch it now, not per variant.
        _silent = np.zeros(8000, dtype=np.float32)
        predict(_silent, 8000, onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=58)
        _BASIC_PITCH_PREDICT = predict
        print("[gate] basic-pitch: enabled", file=sys.stderr)
    except Exception as e:
        print(
            f"[gate] basic-pitch: DISABLED for this run (polyphony check skipped). "
            f"Reason: {e}",
            file=sys.stderr,
        )


def gate_polyphony(mono: np.ndarray, sr: int, pitch_envelope_hz: np.ndarray, frame_rate: float
                   ) -> tuple[Optional[str], str]:
    """Returns (reject_reason, polyphony_check_label).

    Vibrato detection from the CREPE pitch envelope short-circuits this
    gate — vibrato samples otherwise read as polyphonic because Basic
    Pitch fragments vibrato'd notes into multiple short overlapping ones.
    """
    if _detect_vibrato(pitch_envelope_hz, frame_rate):
        return None, "vibrato_bypass"

    _probe_basic_pitch()
    if _BASIC_PITCH_PREDICT is None:
        # Cached unavailable — silent fast path.
        return None, "polyphony_skipped"

    try:
        # Basic Pitch expects a file path or a numpy array; passing the array
        # avoids a temp-file round trip.
        _model_output, note_events, _midi = _BASIC_PITCH_PREDICT(
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
        y, sr = sf.read(str(wav_path), always_2d=True)
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
    # NOTE: only skip_pitch_shift (FX) bypasses pitch MEASUREMENT entirely.
    # A high tolerance (9999) means "don't REJECT on wrong pitch" but we still
    # need to MEASURE the pitch so enrich can snap each sample to the nearest
    # integer semitone. gate_pitch() honors the tolerance internally:
    # `abs(cents) > tolerance_cents` is the wrong_pitch condition, which is
    # unreachable when tolerance is 9999. (Pre-2026-05-22: this branch also
    # short-circuited on tolerance >= 9999, killing enrich's pitch correction.)
    if cfg.skip_pitch_shift:
        # FX: no pitch handling at all
        measured_midi, cents, confidence = float("nan"), 0.0, 1.0
        pitch_env = np.array([])
    else:
        pitch_reason, measured_midi, cents, confidence, pitch_env = gate_pitch(
            mono, sr,
            target_midi=target_pitch,
            tolerance_cents=cfg.pitch_tolerance_cents,
            pitch_floor_hz=cfg.pitch_detection_floor_hz,
            metrics_out=verdict["metrics"],
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
        # Clear any stale failure marker from an earlier round (retry now passes).
        stale = failures_dir / f"{base_id}.json"
        if stale.exists():
            stale.unlink()
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
