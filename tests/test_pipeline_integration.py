"""CPU-only integration tests for the sample-generation pipeline (no model / GPU).

Guards the wiring that a generation run depends on:
  - every enabled category has a config + a prompt file (and vice-versa sanity)
  - pitched roots + drum profiles are well-formed
  - BOTH pipelines normalize PERCEIVED loudness (LUFS), not just peak
  - the retry-to-target helpers (failed_ids / filter_rows)
  - batch_generate work-list expansion (variants, offset, skip-existing) + pre-gate
  - pitch_report aggregation
  - list_to_jsonl roundtrip (drum + multi-source pitched) incl. the `variants` field
  - the deterministic pack builder (smoke)

Run standalone:  python tests/test_pipeline_integration.py
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import batch_generate as BG          # noqa: E402
import run_retry as RR               # noqa: E402
import pitch_report as PR            # noqa: E402
import build_pack as BP              # noqa: E402
import enrich_pitched as EP          # noqa: E402
# NB: postprocess_oneshots is NOT imported (it pulls in tqdm, a pod-only dep).
# Its LUFS defaults are checked by reading the source text instead — keeps this
# suite runnable on a bare Mac.
from pitched_category_config import PITCHED_CATEGORIES  # noqa: E402
from category_config import CATEGORY_NEGATIVES, CATEGORY_DURATIONS  # noqa: E402
from drum_gate_config import DRUM_PROFILES  # noqa: E402


def _read_enabled(path: Path) -> list[str]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def _prompt_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


# ---------------------------------------------------------------- consistency

def test_pitched_enable_list_consistency():
    for cat in _read_enabled(REPO / "scripts/pitched_categories.txt"):
        assert cat in PITCHED_CATEGORIES, f"enabled pitched '{cat}' missing from config"
        assert _prompt_lines(REPO / "prompts/pitched" / f"{cat}.txt") > 0, \
            f"enabled pitched '{cat}' has no prompts"


def test_drum_enable_list_consistency():
    for cat in _read_enabled(REPO / "scripts/categories.txt"):
        assert cat in DRUM_PROFILES, f"enabled drum '{cat}' has no gate profile"
        assert cat in CATEGORY_NEGATIVES, f"enabled drum '{cat}' has no negative prompt"
        assert cat in CATEGORY_DURATIONS, f"enabled drum '{cat}' has no duration"
        assert _prompt_lines(REPO / "prompts" / f"{cat}.txt") > 0, \
            f"enabled drum '{cat}' has no prompts"


def test_pitched_roots_and_zoning_valid():
    for cat, cfg in PITCHED_CATEGORIES.items():
        assert cfg.target_pitches_midi, f"{cat}: no target pitches"
        assert all(0 <= m <= 127 for m in cfg.target_pitches_midi), f"{cat}: root out of MIDI range"
        assert cfg.zone_span_semitones >= 0 and cfg.zone_step_semitones >= 1, f"{cat}: bad zone span/step"
        assert cfg.variants_per_prompt >= 1, f"{cat}: variants<1"


def test_drum_profiles_valid():
    for cat, p in DRUM_PROFILES.items():
        lo, hi = p.centroid_hz
        assert 0 < lo < hi, f"{cat}: bad centroid band {p.centroid_hz}"
        assert p.variants_per_prompt >= 1, f"{cat}: variants<1"
        assert p.kind in ("percussive", "cymbal", "sub", "fx"), f"{cat}: bad kind {p.kind}"


# ------------------------------------------------------------- normalization

def test_both_pipelines_normalize_perceived_loudness():
    """Instruments and drums must both LUFS-normalize (BS.1770 perceived
    loudness), not just peak."""
    # Instruments: enrich targets -20 LUFS.
    assert hasattr(EP, "normalize_lufs"), "enrich missing normalize_lufs"
    assert EP.TARGET_LUFS == -20.0, f"instrument LUFS target changed: {EP.TARGET_LUFS}"
    # Drums: postprocess defaults to LUFS mode at -16 (source-level defaults).
    src = (REPO / "scripts/postprocess_oneshots.py").read_text(encoding="utf-8")
    assert "def normalize_lufs" in src and "def measure_lufs" in src, "postprocess missing LUFS fns"
    assert 'default="lufs"' in src, "drum postprocess default is not LUFS"
    assert "default=-16.0" in src, "drum LUFS target changed from -16"


# -------------------------------------------------------------- retry helpers

def test_retry_assess_target_logic():
    """assess() counts distinct surviving PROMPTS (a multi-source pitched prompt
    survives if ANY of its pitch-ids passed) and returns the failed rows to re-roll."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)  # acts as outputs-dir
        # 2 prompts, each with 2 pitch-ids (multi-source pitched shape).
        src = tmp / "_src.jsonl"
        rows = [
            {"id": "b-p1a", "prompt": "P1"}, {"id": "b-p1b", "prompt": "P1"},
            {"id": "b-p2a", "prompt": "P2"}, {"id": "b-p2b", "prompt": "P2"},
        ]
        src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        gated = tmp / "gated" / "basses"; gated.mkdir(parents=True)
        (gated / "b-p1a.gate.json").write_text("{}")  # P1 survives via one pitch; P2 none

        surv, failed_rows, failed_ids, pool = RR.assess("pitched", tmp, "basses", src)
        assert pool == 2, pool                       # 2 distinct prompts
        assert surv == 1, surv                        # P1 survived (one pitch is enough)
        assert failed_ids == {"b-p2a", "b-p2b"}, failed_ids   # re-roll BOTH of P2's pitches
        assert len(failed_rows) == 2

        # rows_with_new_ids: simulate a top-up that adds P3.
        (src).write_text(src.read_text() + json.dumps({"id": "b-p3a", "prompt": "P3"}) + "\n")
        new_rows, new_ids = RR.rows_with_new_ids(src, {"b-p1a", "b-p1b", "b-p2a", "b-p2b"})
        assert new_ids == {"b-p3a"} and len(new_rows) == 1


# ------------------------------------------------------ batch_generate work-list

def test_build_units_variants_offset_skip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        jl = tmp / "kick.jsonl"
        jl.write_text(json.dumps({"id": "kick-a", "category": "kick", "prompt": "p",
                                  "negative_prompt": "n", "seed": 10, "duration": 1.5,
                                  "variants": 3}) + "\n")
        args = argparse.Namespace(default_duration=1.5, negative_prompt="n",
                                  num_waveforms_per_prompt=1, skip_existing=True,
                                  init_audio_anchor=False, variant_offset=0)
        out_dir, _, units = BG.build_units(jl, tmp / "raw", args)
        assert len(units) == 3 and {u["wav_path"].name for u in units} == \
            {"kick-a_v00.wav", "kick-a_v01.wav", "kick-a_v02.wav"}
        # retry round: offset 64 -> fresh, non-colliding variant indices
        args.variant_offset = 64
        _, _, units2 = BG.build_units(jl, tmp / "raw", args)
        assert {u["variant_index"] for u in units2} == {64, 65, 66}, units2
        assert all(u["wav_path"].name.startswith("kick-a_v6") for u in units2)
        # skip-existing drops already-rendered units
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kick-a_v00.wav").write_bytes(b"x")
        args.variant_offset = 0
        _, _, units3 = BG.build_units(jl, tmp / "raw", args)
        assert len(units3) == 2


def test_pregate_reason():
    good = (0.3 * np.random.RandomState(0).standard_normal((44100, 2))).astype(np.float32)
    assert BG.pregate_reason(good) is None
    nan = good.copy(); nan[0, 0] = np.nan
    assert BG.pregate_reason(nan) == "nan_or_inf"
    assert BG.pregate_reason(np.zeros((44100, 2), np.float32)) == "silent"
    assert BG.pregate_reason(np.ones((44100, 2), np.float32)) == "clipped"


# ---------------------------------------------------------------- pitch report

def test_pitch_report_aggregation():
    with tempfile.TemporaryDirectory() as d:
        gated = Path(d) / "basses"
        (gated / "_failures").mkdir(parents=True)
        # one on-pitch winner, one a semitone off
        (gated / "p1.gate.json").write_text(json.dumps({
            "target_pitch_midi": 40,
            "winner": {"metrics": {"measured_pitch_midi": 40.0, "measured_pitch_cents_offset": 0.0,
                                   "pitch_confidence": 0.9}},
            "all_variants": []}))
        (gated / "p2.gate.json").write_text(json.dumps({
            "target_pitch_midi": 40,
            "winner": {"metrics": {"measured_pitch_midi": 41.0, "measured_pitch_cents_offset": 100.0,
                                   "pitch_confidence": 0.8}},
            "all_variants": []}))
        (gated / "_failures" / "p3.json").write_text(json.dumps({"rejection_reasons": {"unconfident": 2}}))
        stats = PR.analyze_category("basses", gated)
        assert stats["winners"] == 2 and stats["failures"] == 1
        assert abs(stats["pitch_class_accuracy"] - 0.5) < 1e-9  # 1 of 2 on-class
        assert abs(stats["survival_rate"] - (2 / 3)) < 1e-9


# ----------------------------------------------------------- list_to_jsonl roundtrip

def test_list_to_jsonl_pitched_multisource_roundtrip():
    # basses is multi-source (3 roots) -> --limit 2 prompts => 6 rows, all with variants.
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "b.jsonl"
        subprocess.run([sys.executable, str(REPO / "scripts/list_to_jsonl_pitched.py"),
                        "--in", str(REPO / "prompts/pitched/basses.txt"),
                        "--out", str(out), "--limit", "2"], check=True,
                       capture_output=True)
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        pitches = sorted({r["target_pitch_midi"] for r in rows})
        assert len(rows) == 2 * len(pitches), f"expected 2 prompts x {len(pitches)} pitches, got {len(rows)}"
        assert all("variants" in r for r in rows), "pitched rows missing 'variants'"


def test_list_to_jsonl_drum_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "k.jsonl"
        subprocess.run([sys.executable, str(REPO / "scripts/list_to_jsonl.py"),
                        "--in", str(REPO / "prompts/kick.txt"),
                        "--out", str(out), "--limit", "3"], check=True, capture_output=True)
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(rows) == 3 and all("variants" in r for r in rows)


# ------------------------------------------------------------------ pack build

def test_build_pack_smoke():
    BP.smoke_test()  # asserts determinism + multi-source manifest invariants internally


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
