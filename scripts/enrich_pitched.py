"""Take gate-passed pitched samples and bake them into multi-zone instruments.

The gated stage emits one winner per (prompt × target_pitch). This stage groups
all the surviving target pitches of a single PROMPT into ONE multi-source
instrument (so a wide instrument is captured at 2-3 real pitches and each zone
is rendered from its NEAREST real source — small pitch shifts, accurate timbre).
Single-source categories (one target pitch) reduce to the obvious case.

For each instrument it:

  1. Reads each source's gate measured-pitch + variant metadata.
  2. Smart pitch-corrects each source to a clean integer MIDI root: shift to the
     prompted target if measured was within max_correction_semitones, else snap
     to the nearest semitone (≤50¢, no audible artifact). That root is the
     source's "effective root".
  3. LUFS-normalizes to -20 LUFS integrated, -1.0 dBTP ceiling (BS.1770-4).
  4. For sustaining categories (open_ended=True) trims the source to its
     steady-state plateau (Tracktion's SamplerPlugin has no loop API; this is
     the v1 substitute, replayed for the note-hold duration).
  5. Pre-renders pitch-shifted zones on a global stepped grid spanning all
     sources ± zone_span, each zone rendered from its nearest real source via
     RubberBand R3 (formant-preserving). Zones are **16-bit WAV** — memory-
     mappable by Tracktion's SamplerPlugin (no message-thread FLAC decode stall;
     see PluginManager FLAC->WAV transcode note). Sources stay 24-bit WAV.
  6. Writes manifest.json (v1 schema). The plugin consumes `zones[]`; `sources[]`
     is metadata. Zones are guaranteed disjoint + ordered low->high (overlaps
     double-trigger in Tracktion) — asserted before write.

Output:

    outputs/instruments/<cat>/<instrument-id>/
        ├── sources/<root>.wav        (24-bit, one per real source pitch)
        ├── zones/<midi>.wav          (16-bit, nearest-source pitch-shifted)
        ├── manifest.json
        └── prompt.txt

Runs on local CPU (RubberBand is single-threaded but cheap). Reads
outputs/gated/ plus the category JSONL (id -> prompt, for sibling grouping).

Usage:

    python scripts/enrich_pitched.py --category plucks
    python scripts/enrich_pitched.py --category basses --jsonl prompts/pitched/basses.jsonl
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitched_category_config import PITCHED_CATEGORIES, PitchedCategoryConfig


GATE_VERSION = "1.0.0"
ENRICH_VERSION = "2.0.0"  # 2.0 = multi-source + 16-bit WAV zones

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


def instrument_id_for(category_id: str, prompt: str) -> str:
    """Stable, pitch-independent instrument id from the prompt, so all target
    pitches of one prompt land in the SAME instrument folder. Normalized like
    list_to_jsonl's dedup key so wording-equivalent prompts collapse."""
    norm = " ".join(prompt.lower().split())
    h = hashlib.sha1(f"{category_id}:{norm}".encode("utf-8")).hexdigest()[:8]
    return f"{category_id}-{h}"


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
# Per-source preparation (smart pitch-correct + trim + normalize)
# -----------------------------------------------------------------------------

def correct_and_prepare_source(y: np.ndarray, sr: int, target_midi: int, metrics: dict,
                               cfg: PitchedCategoryConfig) -> tuple[np.ndarray, int, int, dict]:
    """Smart pitch-correct one source to a clean integer MIDI root, optionally
    sustain-trim, then LUFS-normalize. Returns
    (y_norm, effective_root_midi, correction_applied_cents, loudness_meta).

    Correction policy (unchanged from v1): measured within max_correction_semitones
    of target -> shift to target (preserves prompt semantics); else snap to the
    nearest integer semitone (≤50¢, no audible artifact).
    """
    cents_offset = metrics.get("measured_pitch_cents_offset")
    measured_midi_metric = metrics.get("measured_pitch_midi")
    effective_root_midi = target_midi
    correction_applied = 0

    if cfg.skip_pitch_shift:
        pass
    elif measured_midi_metric is not None and math.isfinite(float(measured_midi_metric)):
        measured_midi = float(measured_midi_metric)
        distance_semitones = abs(measured_midi - target_midi)
        if distance_semitones <= cfg.max_correction_semitones:
            effective_root_midi = target_midi
        else:
            effective_root_midi = int(round(measured_midi))
        shift_semitones = effective_root_midi - measured_midi
        if abs(shift_semitones) > 0.05:  # > 5 cents — worth a shift
            y = pitch_shift(y, sr, shift_semitones, preserve_formants=True)
            correction_applied = int(round(shift_semitones * 100.0))
    elif cents_offset is not None and abs(cents_offset) > 5.0:
        # Fallback for older gate verdicts: cents_offset only, no measured midi.
        cap_cents = cfg.max_correction_semitones * 100
        clipped_cents = max(-cap_cents, min(cap_cents, cents_offset))
        semitones = -clipped_cents / 100.0
        y = pitch_shift(y, sr, semitones, preserve_formants=True)
        correction_applied = int(round(semitones * 100.0))
        effective_root_midi = target_midi

    if cfg.open_ended:
        y = trim_to_sustain(y, sr, cfg.min_sustain_seconds, pad_seconds=0.5)

    y_norm, loud_meta = normalize_lufs(y, sr)
    return y_norm, int(effective_root_midi), correction_applied, loud_meta


# -----------------------------------------------------------------------------
# Zone planning (global stepped grid, nearest-source assignment, disjoint)
# -----------------------------------------------------------------------------

def _build_zone_plan(source_roots: list[int], zone_span: int, zone_step: int) -> list[dict]:
    """Plan a disjoint, ordered set of key zones across all sources.

    Roots = each source's exact effective root (rendered unshifted) PLUS a
    stepped grid spanning [min(sources)-span, max(sources)+span]. Each root is
    assigned to its NEAREST source (tie -> lower-pitched source), so every
    zone's pitch shift is small. min/max are computed by midpoint-splitting
    adjacent roots, guaranteeing contiguous 0..127 coverage with no overlap.

    Returns list of {root, src_idx, min_midi, max_midi} ordered low->high.
    """
    if not source_roots:
        return []
    roots: set[int] = set(int(r) for r in source_roots)  # always render exact source roots unshifted
    lo = max(0, min(source_roots) - zone_span)
    hi = min(127, max(source_roots) + zone_span)
    for r in range(lo, hi + 1, zone_step):
        roots.add(r)
    ordered = sorted(roots)

    plan: list[dict] = []
    for i, root in enumerate(ordered):
        src_idx = min(range(len(source_roots)),
                      key=lambda k: (abs(source_roots[k] - root), source_roots[k]))
        min_midi = 0 if i == 0 else (ordered[i - 1] + root) // 2 + 1
        max_midi = 127 if i == len(ordered) - 1 else (root + ordered[i + 1]) // 2
        plan.append({"root": root, "src_idx": src_idx, "min_midi": min_midi, "max_midi": max_midi})

    # Disjoint + ordered + full-coverage invariant (overlaps double-trigger in Tracktion).
    prev_max = -1
    for z in plan:
        assert z["min_midi"] <= z["root"] <= z["max_midi"], f"root outside its zone: {z}"
        assert z["min_midi"] > prev_max, f"zone overlap: {z} (prev_max={prev_max})"
        prev_max = z["max_midi"]
    assert plan[0]["min_midi"] == 0 and plan[-1]["max_midi"] == 127, "zones must cover 0..127"
    return plan


# -----------------------------------------------------------------------------
# Per-instrument enrich (1..N real sources -> one instrument folder)
# -----------------------------------------------------------------------------

def enrich_instrument(items: list[dict], category_id: str, cfg: PitchedCategoryConfig,
                      category_out_dir: Path, raw_meta_dir: Optional[Path]) -> Optional[Path]:
    """Build ONE instrument folder from a group of gated sources (the surviving
    target pitches of a single prompt). Returns the manifest path, or None if no
    source WAV was usable."""
    prepared: list[dict] = []
    for it in items:
        gate_wav = Path(it["gate_wav"])
        if not gate_wav.exists():
            continue
        gate_report = it["gate_report"]
        winner = gate_report.get("winner", {}) or {}
        metrics = winner.get("metrics", {}) or {}
        target_midi = int(it["target_pitch"]) if it.get("target_pitch") is not None \
            else int(gate_report.get("target_pitch_midi", 60))
        try:
            y, sr = sf.read(str(gate_wav), always_2d=True)
        except Exception as e:
            print(f"[enrich] read failed {gate_wav}: {e}", file=sys.stderr)
            continue

        y_norm, eff_root, corr, loud_meta = correct_and_prepare_source(y, sr, target_midi, metrics, cfg)

        variant_index = int(winner.get("variant_index", 0))
        suffix = f"_v{variant_index:02d}" if variant_index >= 0 else ""
        raw_meta: dict = {}
        if raw_meta_dir is not None:
            rmp = raw_meta_dir / f"{it['id']}{suffix}.json"
            if rmp.exists():
                try:
                    raw_meta = json.loads(rmp.read_text(encoding="utf-8"))
                except Exception:
                    raw_meta = {}

        prepared.append({
            "y": y_norm, "sr": sr, "eff_root": eff_root, "corr": corr,
            "loud_meta": loud_meta, "metrics": metrics, "target_midi": target_midi,
            "variant_index": variant_index, "raw_meta": raw_meta,
            "raw_winner_path": gate_report.get("raw_winner_path"),
            "conf": metrics.get("pitch_confidence") or 0.0,
        })

    if not prepared:
        return None

    # Dedup sources that resolved to the same effective root (keep best confidence),
    # then order low->high so sources/ and zones/ list by pitch.
    by_root: dict[int, dict] = {}
    for p in prepared:
        r = p["eff_root"]
        if r not in by_root or (p["conf"] or 0) > (by_root[r]["conf"] or 0):
            by_root[r] = p
    prepared = [by_root[r] for r in sorted(by_root)]

    sr = prepared[0]["sr"]
    prompt = next((it["prompt"] for it in items if it.get("prompt")), "") \
        or next((p["raw_meta"].get("prompt", "") for p in prepared if p["raw_meta"].get("prompt")), "")
    instrument_id = instrument_id_for(category_id, prompt) if prompt else (items[0].get("id") or f"{category_id}-unknown")

    inst_dir = category_out_dir / instrument_id
    sources_dir = inst_dir / "sources"
    zones_dir = inst_dir / "zones"
    sources_dir.mkdir(parents=True, exist_ok=True)
    zones_dir.mkdir(parents=True, exist_ok=True)

    # Write source WAVs (24-bit master).
    source_entries: list[dict] = []
    for p in prepared:
        sfname = f"{midi_to_filename(p['eff_root'])}.wav"
        sf.write(str(sources_dir / sfname), p["y"], sr, subtype="PCM_24")
        m, lm, rm = p["metrics"], p["loud_meta"], p["raw_meta"]
        source_entries.append({
            "file": f"sources/{sfname}",
            "target_pitch_midi": p["target_midi"],
            "measured_pitch_midi": m.get("measured_pitch_midi"),
            "measured_pitch_cents_offset": m.get("measured_pitch_cents_offset"),
            "pitch_confidence": m.get("pitch_confidence"),
            "pitch_correction_applied_cents": p["corr"],
            "polyphony_check": m.get("polyphony_check"),
            "onset_ms": m.get("onset_ms"),
            "sustain_quality": m.get("sustain_quality"),
            "loudness_lufs": lm.get("loudness_lufs_out"),
            "rms_dbfs": lm.get("rms_dbfs"),
            "duration_seconds": round(p["y"].shape[0] / sr, 3),
            "raw_path": p["raw_winner_path"],
            "seed": rm.get("seed"),
            "variant_index": p["variant_index"],
            "effective_root_midi": p["eff_root"],
        })

    # Pre-render zones as 16-bit WAV (mmap-able by Tracktion's SamplerPlugin).
    zones: list[dict] = []
    if cfg.skip_pitch_shift:
        # Sound-design content (legacy fx): a single zone across the whole keyboard.
        p = prepared[0]
        zfname = f"{midi_to_filename(p['eff_root'])}.wav"
        sf.write(str(zones_dir / zfname), p["y"], sr, subtype="PCM_16")
        zones.append({"sample": f"zones/{zfname}", "root_midi": p["eff_root"], "min_midi": 0, "max_midi": 127})
    else:
        source_roots = [p["eff_root"] for p in prepared]
        plan = _build_zone_plan(source_roots, cfg.zone_span_semitones, cfg.zone_step_semitones)
        for z in plan:
            root, src = z["root"], prepared[z["src_idx"]]
            zfname = f"{midi_to_filename(root)}.wav"
            shift = root - src["eff_root"]
            data = src["y"] if shift == 0 else pitch_shift(src["y"], sr, shift, preserve_formants=True)
            sf.write(str(zones_dir / zfname), data, sr, subtype="PCM_16")
            zones.append({"sample": f"zones/{zfname}", "root_midi": root,
                          "min_midi": z["min_midi"], "max_midi": z["max_midi"]})

    if prompt:
        (inst_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    rm0 = prepared[0]["raw_meta"]
    channels = int(prepared[0]["y"].shape[1]) if prepared[0]["y"].ndim == 2 else 1
    manifest = {
        "schema_version": 1,
        "instrument_id": instrument_id,
        "category_id": category_id,
        "category_display": category_id.replace("-", " ").title(),
        "prompt": prompt,
        "negative_prompt": rm0.get("negative_prompt", ""),
        "model": rm0.get("model", "stabilityai/stable-audio-3-medium"),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "gate_version": GATE_VERSION,
        "enrich_version": ENRICH_VERSION,
        "sources": source_entries,
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


# -----------------------------------------------------------------------------
# Main: group gated winners by prompt, enrich each group in parallel
# -----------------------------------------------------------------------------

def _enrich_one(args: tuple) -> tuple[str, str, Optional[int], Optional[str]]:
    """Worker entry point for ProcessPoolExecutor. Returns
    (label, status, zone_count, err); status is 'ok' / 'skipped' / 'failed'."""
    key, items, category_id, cfg, out_dir, raw_meta_dir = args
    try:
        manifest_path = enrich_instrument(
            items, category_id, cfg, Path(out_dir), Path(raw_meta_dir) if raw_meta_dir else None,
        )
        if manifest_path is None:
            return (str(key)[:48], "skipped", None, "no usable source wav")
        zone_count = len(json.loads(Path(manifest_path).read_text())["zones"])
        return (Path(manifest_path).parent.name, "ok", zone_count, None)
    except Exception as e:
        return (str(key)[:48], "failed", None, str(e))


def run_enrich(category_id: str, cfg: PitchedCategoryConfig, in_dir: Path, out_dir: Path,
               raw_meta_dir: Optional[Path], jsonl_path: Optional[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_jsons = sorted(in_dir.glob("*.gate.json"))
    if not gate_jsons:
        print(f"[enrich] no *.gate.json under {in_dir}; nothing to do")
        return

    # id -> prompt, so the surviving target pitches of one prompt merge into one
    # multi-source instrument. Without the JSONL we can't group siblings; fall
    # back to one instrument per gated source (correct, just not merged).
    prompt_by_id: dict[str, str] = {}
    if jsonl_path and Path(jsonl_path).exists():
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_by_id[row["id"]] = row.get("prompt", "")
    else:
        print(f"[enrich] WARNING: jsonl {jsonl_path} not found — multi-source siblings "
              f"will NOT be merged (one instrument per gated source)", file=sys.stderr)

    groups: dict[str, list[dict]] = defaultdict(list)
    for gj in gate_jsons:
        gate_wav = gj.with_suffix("").with_suffix(".wav")  # strip ".gate.json"
        try:
            gate_report = json.loads(gj.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[enrich] bad gate.json {gj.name}: {e}", file=sys.stderr)
            continue
        gid = gate_report.get("id") or gj.name[: -len(".gate.json")]
        prompt = prompt_by_id.get(gid)
        key = prompt if prompt else f"__id__{gid}"  # fallback: no merge
        groups[key].append({
            "gate_wav": str(gate_wav),
            "gate_report": gate_report,
            "target_pitch": gate_report.get("target_pitch_midi"),
            "prompt": prompt or "",
            "id": gid,
        })

    workers_env = os.environ.get("WORKERS")
    if workers_env:
        try:
            workers = max(1, int(workers_env))
        except ValueError:
            workers = max(1, (os.cpu_count() or 2) - 1)
    else:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(groups))
    print(f"[enrich] {len(groups)} instruments from {len(gate_jsons)} gated sources; {workers} workers")

    tasks: list[tuple] = [
        (key, items, category_id, cfg, str(out_dir), str(raw_meta_dir) if raw_meta_dir else None)
        for key, items in groups.items()
    ]

    enriched = 0
    skipped = 0

    if workers == 1:
        for t in tasks:
            label, status, zones, err = _enrich_one(t)
            if status == "ok":
                enriched += 1
                print(f"[enrich] {label}: ok ({zones} zones)")
            else:
                skipped += 1
                print(f"[enrich] {label}: {status.upper()} — {err}", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_enrich_one, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    label, status, zones, err = fut.result()
                except Exception as e:  # noqa: BLE001
                    skipped += 1
                    print(f"[enrich] worker crashed: {e}", file=sys.stderr)
                    continue
                if status == "ok":
                    enriched += 1
                    print(f"[enrich] {label}: ok ({zones} zones)", flush=True)
                else:
                    skipped += 1
                    print(f"[enrich] {label}: {status.upper()} — {err}", file=sys.stderr, flush=True)

    print(f"[enrich] done. enriched={enriched} skipped={skipped}")


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
    parser.add_argument("--jsonl", default=None,
                        help="Category JSONL (id -> prompt, for sibling-pitch grouping). "
                             "Default: prompts/pitched/<cat>.jsonl")
    parser.add_argument("--outputs-dir", default=None,
                        help="Override $SAS_OUTPUTS_DIR")
    args = parser.parse_args()

    cfg = PITCHED_CATEGORIES.get(args.category)
    if cfg is None:
        sys.exit(f"unknown pitched category {args.category!r}")

    repo_root = Path(__file__).resolve().parent.parent
    outputs_dir = Path(args.outputs_dir or os.environ.get("SAS_OUTPUTS_DIR", "outputs"))
    in_dir = Path(args.in_dir) if args.in_dir else outputs_dir / "gated" / args.category
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir / "instruments" / args.category
    raw_meta_dir = Path(args.raw_meta_dir) if args.raw_meta_dir else outputs_dir / "raw" / args.category / "_metadata"
    if not raw_meta_dir.exists():
        raw_meta_dir = None
    jsonl_path = Path(args.jsonl) if args.jsonl else repo_root / "prompts" / "pitched" / f"{args.category}.jsonl"

    run_enrich(args.category, cfg, in_dir, out_dir, raw_meta_dir, jsonl_path)


if __name__ == "__main__":
    main()
