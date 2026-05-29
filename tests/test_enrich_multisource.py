"""Unit test for the multi-source enrich merge (enrich_pitched.enrich_instrument).

Two gated source pitches of one prompt must merge into ONE instrument with two
24-bit WAV sources and a disjoint, ordered set of 16-bit WAV zones covering
0..127. Needs the `rubberband` CLI for the actual pitch shift; the structural
assertions hold regardless (pitch_shift no-ops to a copy if rubberband is
missing), so the test runs anywhere.

Run standalone:  python tests/test_enrich_multisource.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import enrich_pitched as E  # noqa: E402
from pitched_category_config import PITCHED_CATEGORIES  # noqa: E402

SR = 44100


def _tone(midi, dur=2.0):
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(SR * dur)) / SR
    y = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
    r = int(0.02 * SR)
    env = np.ones_like(t); env[:r] = np.linspace(0, 1, r); env[-r:] = np.linspace(1, 0, r)
    return (0.8 * y / np.max(np.abs(y)) * env).astype(np.float32)


def _gate_report(sid, target):
    return {
        "id": sid, "target_pitch_midi": target,
        "winner": {"passed": True, "variant_index": 0, "score": 0.8,
                   "metrics": {"measured_pitch_midi": float(target),
                               "measured_pitch_cents_offset": 0.0,
                               "pitch_confidence": 0.95, "sustain_quality": 0.9}},
        "all_variants": [], "raw_winner_path": f"/raw/{sid}_v00.wav",
    }


def _assert_zone_invariants(zones):
    prev = -1
    for z in zones:
        assert z["min_midi"] <= z["root_midi"] <= z["max_midi"], f"root outside zone {z}"
        assert z["min_midi"] > prev, f"overlap/out-of-order {z} prev={prev}"
        assert z["sample"].endswith(".wav"), f"zone not WAV: {z['sample']}"
        prev = z["max_midi"]
    assert zones[0]["min_midi"] == 0 and zones[-1]["max_midi"] == 127, "must cover 0..127"


def test_two_sources_merge_into_one_instrument():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        gdir = tmp / "gated" / "basses"
        gdir.mkdir(parents=True)
        items = []
        for sid, tp in [("basses-aaaa", 40), ("basses-bbbb", 52)]:
            sf.write(str(gdir / f"{sid}.wav"), _tone(tp), SR, subtype="PCM_24")
            items.append({"gate_wav": str(gdir / f"{sid}.wav"),
                          "gate_report": _gate_report(sid, tp),
                          "target_pitch": tp, "prompt": "warm sub bass single note",
                          "id": sid})
        out = tmp / "instruments" / "basses"
        mp = E.enrich_instrument(items, "basses", PITCHED_CATEGORIES["basses"], out, None)
        assert mp is not None, "enrich produced no instrument"
        m = json.loads(Path(mp).read_text())

        assert m["schema_version"] == 1
        assert len(m["sources"]) == 2, f"expected 2 sources, got {len(m['sources'])}"
        _assert_zone_invariants(m["zones"])

        inst = Path(mp).parent
        # zones are 16-bit WAV (mmap-able), sources are 24-bit WAV
        z0 = inst / m["zones"][0]["sample"]
        assert sf.info(str(z0)).subtype == "PCM_16", "zones must be 16-bit WAV"
        s0 = inst / m["sources"][0]["file"]
        assert sf.info(str(s0)).subtype == "PCM_24", "sources must be 24-bit WAV"
        # all source + zone files referenced actually exist
        for s in m["sources"]:
            assert (inst / s["file"]).exists(), f"missing source {s['file']}"
        for z in m["zones"]:
            assert (inst / z["sample"]).exists(), f"missing zone {z['sample']}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
