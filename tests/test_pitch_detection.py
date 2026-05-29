"""Unit tests for the spectral fundamental detector in gate_pitched.py.

These cover the pitch NUMBER logic only — torchcrepe (voicing/confidence) runs
on the GPU pod and is not exercised here. The detector is pure NumPy, so these
run anywhere (CI, laptop) with no torch / CUDA.

Run standalone (no pytest needed):   python tests/test_pitch_detection.py
Or under pytest:                      pytest tests/test_pitch_detection.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gate_pitched import (  # noqa: E402
    spectral_fundamental_midi,
    ensemble_octave_check,
    _harmonic_sum_score,
    _avg_magnitude_spectrum,
    midi_to_hz,
)

SR = 44100


def _synth(midi: float, partials: dict[int, float], dur: float = 2.0,
           sr: int = SR) -> np.ndarray:
    """Synthesize a sustained tone at `midi` with the given harmonic amplitudes
    ({harmonic_number: amplitude}). A short raised-cosine attack/decay avoids
    click transients that would dirty the onset region."""
    f0 = midi_to_hz(midi)
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t)
    for h, amp in partials.items():
        y += amp * np.sin(2.0 * np.pi * f0 * h * t)
    env = np.ones_like(t)
    ramp = int(0.02 * sr)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    y *= env
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_pure_tone_returns_target():
    """A near-pure C3 resolves to ~C3 (MIDI 48)."""
    y = _synth(48, {1: 1.0, 2: 0.15})
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert abs(got - 48) < 0.6, got


def test_harmonic_rich_not_a_fourth_sharp():
    """Regression for the shipped bug: a harmonically-rich C3 (fundamental plus
    strong 2nd–6th partials) must NOT read as ~F3 (MIDI 53). torchcrepe's median
    settled on the phantom value between C3 and its partials; the spectral
    fundamental must land on C3."""
    y = _synth(48, {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.6, 5: 0.5, 6: 0.4})
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert abs(got - 48) < 0.6, f"expected ~48 (C3), got {got} — the 4th-sharp bug"
    assert abs(got - 53) > 2.0, f"must not read as F3, got {got}"


def test_louder_second_harmonic_keeps_pitch_class():
    """Octave ambiguity: a tone whose 2nd harmonic is LOUDER than its
    fundamental (a C2 fundamental presenting strongest at C3) must still resolve
    to pitch class C, seated near the requested register — not to the louder
    partial's class."""
    y = _synth(36, {1: 0.4, 2: 1.0, 3: 0.5, 4: 0.7})  # C2 fundamental, loud C3
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert round(got) % 12 == 0, f"expected pitch class C, got {got}"
    assert abs(got - 48) <= 12.5, f"expected near requested C3 register, got {got}"


def test_sub_bass_fundamental():
    """Sub-bass (E1, MIDI 40 ≈ 41 Hz) with harmonics resolves to ~E1, replacing
    the old fragile 3-way agreement branch."""
    y = _synth(40, {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.3})
    got = spectral_fundamental_midi(y, SR, target_midi=40)
    assert round(got) % 12 == 4, f"expected pitch class E, got {got}"
    assert abs(got - 40) <= 12.5, got


def test_detuned_tone_reports_cents():
    """A C3 that is +30 cents sharp resolves to ~48.3 — the sub-semitone offset
    is preserved so enrich's fine pitch-correction can land it exactly."""
    y = _synth(48.30, {1: 1.0, 2: 0.3})
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert 48.1 < got < 48.5, got


def test_octave_reseat_to_target_register():
    """A real F (pitch class F) produced far above the requested C3 register is
    re-seated to the F nearest C3, keeping the (correct, in-key) pitch class."""
    y = _synth(65, {1: 1.0, 2: 0.5, 3: 0.4})  # F4
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert round(got) % 12 == 5, f"expected pitch class F, got {got}"
    assert abs(got - 48) <= 7, f"expected F nearest C3 (≈53), got {got}"


def test_harmonic_sum_prefers_fundamental_over_subharmonic():
    """The 1/h-weighted harmonic sum scores the true fundamental above its
    subharmonic (an octave below), so we don't drift an octave low."""
    y = _synth(48, {1: 1.0, 2: 0.8, 3: 0.6})
    mag, freqs = _avg_magnitude_spectrum(y, SR)
    f_fund = midi_to_hz(48)
    f_sub = midi_to_hz(36)  # one octave below
    assert _harmonic_sum_score(mag, freqs, f_fund) > _harmonic_sum_score(mag, freqs, f_sub)


def test_silence_returns_nan():
    """No clear peak (silence) → NaN, so the caller can fall back gracefully."""
    y = np.zeros(SR, dtype=np.float32)
    got = spectral_fundamental_midi(y, SR, target_midi=48)
    assert math.isnan(got), got


# --- ensemble_octave_check (sub-bass autocorr cross-check, pure NumPy) -------

def test_ensemble_rescues_wrong_spectral_class():
    """When spectral mis-reads a clean sub-bass tone by a couple semitones,
    the autocorrelation cross-check (clear period) overrides it back to truth."""
    y = _synth(40, {1: 1.0, 2: 0.5, 3: 0.3})  # E1, MIDI 40
    info: dict = {}
    got = ensemble_octave_check(y, SR, spectral_midi=42.0, target_midi=40,
                                pitch_floor_hz=30.0, info_out=info)
    assert info["ensemble_applied"] is True, info
    assert abs(got - 40) < 1.0, f"expected rescue to ~40, got {got}"


def test_ensemble_keeps_correct_spectral():
    """A correct spectral reading is never overridden (no false rescue)."""
    y = _synth(40, {1: 1.0, 2: 0.5, 3: 0.3})
    info: dict = {}
    got = ensemble_octave_check(y, SR, spectral_midi=40.0, target_midi=40,
                                pitch_floor_hz=30.0, info_out=info)
    assert info["ensemble_applied"] is False, info
    assert got == 40.0, got


def test_ensemble_ignores_noise():
    """Broadband noise has no clear period (low autocorr strength) → the
    spectral value is returned unchanged rather than chasing a phantom pitch."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    info: dict = {}
    got = ensemble_octave_check(noise, SR, spectral_midi=42.0, target_midi=40,
                                pitch_floor_hz=30.0, info_out=info)
    assert info["ensemble_applied"] is False, info
    assert got == 42.0, got
    assert info["autocorr_strength"] < 0.5, info


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
