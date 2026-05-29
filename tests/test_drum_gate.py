"""Unit tests for the drum one-shot quality gate (gate_drums.py).

Synthesizes good + deliberately-bad drum one-shots and asserts the gate accepts
the good ones and rejects the bad (silent / clipped / sustained drone /
multi-hit loop), plus an end-to-end best-of-N selection. Needs librosa (onset /
rms / spectral-centroid); skips cleanly if it's unavailable.

Run standalone:   python tests/test_drum_gate.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gate_drums import evaluate_variant, run_gate, gate_prefilter  # noqa: E402
from drum_gate_config import get_profile  # noqa: E402

try:
    import librosa  # noqa: F401
    _HAVE_LIBROSA = True
except Exception:
    _HAVE_LIBROSA = False

SR = 44100


def _kick(dur=0.35, sr=SR):
    """A clean kick: short high click transient + decaying low body."""
    t = np.arange(int(sr * dur)) / sr
    click = np.exp(-t * 900) * np.sin(2 * np.pi * 2500 * t) * 0.4
    body = np.exp(-t * 12) * np.sin(2 * np.pi * 60 * t)
    y = 0.3 * click + body
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)


def _sustained(dur=1.0, sr=SR):
    """A constant-amplitude low drone — should reject (never decays)."""
    t = np.arange(int(sr * dur)) / sr
    return (0.8 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)


def _two_hits(sr=SR):
    """Two kicks 300 ms apart — a loop/roll, not a one-shot."""
    a = _kick()
    gap = np.zeros(int(sr * 0.3), dtype=np.float32)
    return np.concatenate([a, gap, a])


def _write(tmp: Path, name: str, y: np.ndarray) -> Path:
    p = tmp / name
    sf.write(str(p), y, SR, subtype="PCM_24")
    return p


def _verdict(tmp: Path, y: np.ndarray, cat="kick") -> dict:
    return evaluate_variant(_write(tmp, "x.wav", y), get_profile(cat))


def test_prefilter_silent_and_clipped():
    assert gate_prefilter(np.zeros((SR, 1), np.float32)) == "silent"
    # Zero-mean clipped sine (all-ones would trip dc_offset first).
    t = np.arange(SR) / SR
    clipped = np.clip(np.sin(2 * np.pi * 200 * t) * 3.0, -1.0, 1.0).astype(np.float32)
    assert gate_prefilter(clipped[:, None]) == "clipped"
    assert gate_prefilter(_kick()[:, None]) is None


def test_good_kick_passes():
    if not _HAVE_LIBROSA:
        print("(skip: no librosa)"); return
    with tempfile.TemporaryDirectory() as d:
        v = _verdict(Path(d), _kick())
    assert v["passed"], f"good kick rejected: {v['rejection_reason']} {v['metrics']}"
    assert v["score"] > 0


def test_sustained_drone_rejected():
    if not _HAVE_LIBROSA:
        print("(skip: no librosa)"); return
    with tempfile.TemporaryDirectory() as d:
        v = _verdict(Path(d), _sustained())
    assert not v["passed"] and v["rejection_reason"] == "sustained_no_decay", v


def test_multi_hit_loop_rejected():
    if not _HAVE_LIBROSA:
        print("(skip: no librosa)"); return
    with tempfile.TemporaryDirectory() as d:
        v = _verdict(Path(d), _two_hits())
    assert not v["passed"] and v["rejection_reason"] == "multi_hit_loop", v


def test_best_of_n_selection_end_to_end():
    """run_gate over 3 variants of one prompt: one clean kick, one sustained
    drone (reject), one silent (reject) -> the clean kick wins and is copied."""
    if not _HAVE_LIBROSA:
        print("(skip: no librosa)"); return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        raw = tmp / "raw" / "kick"
        (raw / "_metadata").mkdir(parents=True)
        _write(raw, "kick-aaa_v00.wav", _kick())
        _write(raw, "kick-aaa_v01.wav", _sustained())
        _write(raw, "kick-aaa_v02.wav", np.zeros(int(SR * 0.3), np.float32))
        (raw / "_metadata" / "kick-aaa_v00.json").write_text(
            json.dumps({"id": "kick-aaa", "prompt": "deep kick one shot"}))
        out = tmp / "gated_drums" / "kick"
        run_gate("kick", raw, out)

        winner = out / "kick-aaa.wav"
        assert winner.exists(), "winner WAV not copied"
        assert (out / "kick-aaa.drumgate.json").exists()
        # winner metadata carried across (postprocess reads the prompt from it)
        meta = out / "_metadata" / "kick-aaa.json"
        assert meta.exists() and json.loads(meta.read_text())["prompt"] == "deep kick one shot"
        report = json.loads((out / "kick-aaa.drumgate.json").read_text())
        assert report["winner"]["variant_index"] == 0, "expected the clean kick (v00) to win"


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
