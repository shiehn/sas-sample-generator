"""Post-process generated WAVs: trim silence, normalize, optional mono downmix.

Two invocation styles:

  # Category-aware (recommended): resolves paths under $SAS_OUTPUTS_DIR
  python scripts/postprocess_oneshots.py --category kick --mono

  # Explicit (back-compat for one-offs):
  python scripts/postprocess_oneshots.py \\
    --in-dir outputs/raw/kick \\
    --out-dir outputs/processed/kick \\
    --rejected-dir outputs/rejected/kick \\
    --mono
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


# BS.1770 needs >0.4 s of audio per block and the relative-gate behavior
# stabilizes above ~3 s. 3.5 s leaves >=7 momentary blocks even after gating.
LUFS_MIN_SECONDS = 3.5

# Per-sample prompts ride along as a flat sibling <id>.txt next to <id>.wav
# in the same category dir. Merging two generation runs is then a single
# rsync. The positive prompt is also written into the WAV's RIFF INFO ICMT
# chunk (visible to Logic / Ableton / Audacity / ffprobe) so DAW users get
# the prompt without the .txt companion. Structured generation telemetry
# (seed, model, LUFS) stays upstream on the pod under
# outputs/raw/<cat>/_metadata/<id>.json (written by batch_generate.py).
SOFTWARE_TAG = "sas-sample-generator + Stable Audio 3"


def db_to_amp(db):
    return 10 ** (db / 20)


def ensure_2d(y):
    if y.ndim == 1:
        return y[:, None]
    return y


def apply_fade(y, sr, fade_ms=5):
    fade_len = int(sr * fade_ms / 1000)
    if fade_len <= 1 or len(y) < fade_len * 2:
        return y

    fade_in = np.linspace(0, 1, fade_len)[:, None]
    fade_out = np.linspace(1, 0, fade_len)[:, None]

    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y


def trim_silence(y, sr, threshold_db=-45, pad_ms=15):
    y = ensure_2d(y)
    mono = np.mean(y, axis=1)

    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak < 1e-5:
        return None, "too_quiet"

    threshold = peak * db_to_amp(threshold_db)
    active = np.where(np.abs(mono) >= threshold)[0]

    if len(active) == 0:
        return None, "silence"

    pad = int(sr * pad_ms / 1000)
    start = max(0, int(active[0]) - pad)
    end = min(len(y), int(active[-1]) + pad)

    return y[start:end], None


def normalize_peak(y, target_db=-1.0):
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-5:
        return None
    target = db_to_amp(target_db)
    return np.clip(y / peak * target, -1.0, 1.0)


def compute_loop_padded(y, sr, min_seconds=LUFS_MIN_SECONDS):
    """Tile y until length >= min_seconds. y is 2D (samples, channels)."""
    target = int(np.ceil(min_seconds * sr))
    if len(y) >= target:
        return y
    reps = int(np.ceil(target / max(len(y), 1)))
    return np.tile(y, (reps, 1))[:target]


def measure_lufs(y, sr):
    """Integrated LUFS (BS.1770-4). Loops short samples for measurement.
    Returns -inf when pyloudnorm can't measure (degenerate input)."""
    padded = compute_loop_padded(y, sr)
    buf = padded[:, 0] if padded.shape[1] == 1 else padded
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(buf))
    except Exception:
        return float("-inf")
    return lufs if np.isfinite(lufs) else float("-inf")


def _amp_to_dbfs(a):
    return 20.0 * np.log10(a) if a > 0 else float("-inf")


def read_raw_prompt(wav_path):
    """Read the positive prompt from batch_generate.py's raw sidecar.
    Returns '' if the JSON is missing or unreadable (manual drop, pre-PR run)."""
    meta_path = wav_path.parent / "_metadata" / f"{wav_path.stem}.json"
    if not meta_path.exists():
        return ""
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("prompt", "")
    except (json.JSONDecodeError, OSError):
        return ""


def write_prompt_txt(out_dir, sample_id, prompt):
    """Sibling <id>.txt next to <id>.wav. Always written (empty if no prompt)
    so consumers get a consistent invariant: one .txt per .wav."""
    txt_path = out_dir / f"{sample_id}.txt"
    txt_path.write_text(prompt, encoding="utf-8")
    return txt_path


def normalize_lufs(y, sr, target_lufs, peak_ceiling_db):
    """Apply LUFS gain then hard peak ceiling. Returns (audio_or_None, meta)."""
    meta = {
        "loudness_lufs_in": None,
        "loudness_lufs_out": None,
        "lufs_gain_db": None,
        "peak_dbfs_in": None,
        "peak_dbfs_out": None,
        "peak_ceiling_applied": False,
        "loop_padded_for_measurement": len(y) < int(LUFS_MIN_SECONDS * sr),
        "lufs_unmeasurable": False,
    }
    peak_in = float(np.max(np.abs(y))) if len(y) else 0.0
    meta["peak_dbfs_in"] = _amp_to_dbfs(peak_in)

    lufs_in = measure_lufs(y, sr)
    if not np.isfinite(lufs_in):
        meta["lufs_unmeasurable"] = True
        return None, meta
    meta["loudness_lufs_in"] = lufs_in

    gain_db = target_lufs - lufs_in
    out = y * db_to_amp(gain_db)
    meta["lufs_gain_db"] = gain_db

    ceiling_amp = db_to_amp(peak_ceiling_db)
    peak_post = float(np.max(np.abs(out)))
    if peak_post > ceiling_amp:
        out = out * (ceiling_amp / peak_post)
        meta["peak_ceiling_applied"] = True

    final_peak = float(np.max(np.abs(out)))
    meta["peak_dbfs_out"] = _amp_to_dbfs(final_peak)
    meta["loudness_lufs_out"] = measure_lufs(out, sr)
    return out, meta


def process_file(path, out_dir, rejected_dir, args):
    y, sr = sf.read(str(path), always_2d=True)
    sample_id = path.stem
    raw_prompt = read_raw_prompt(path)

    trimmed, reject_reason = trim_silence(
        y, sr,
        threshold_db=args.trim_threshold_db,
        pad_ms=args.pad_ms,
    )

    if reject_reason:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        write_prompt_txt(rejected_dir, sample_id, raw_prompt)
        return {"status": "rejected", "reason": reject_reason}

    max_samples = int(args.max_seconds * sr)
    if len(trimmed) > max_samples:
        trimmed = trimmed[:max_samples]

    if args.mono:
        trimmed = np.mean(trimmed, axis=1, keepdims=True)

    if args.normalize == "lufs":
        processed, _ = normalize_lufs(
            trimmed, sr, args.target_lufs, args.peak_ceiling_db)
        fail_reason = "lufs_unmeasurable"
    elif args.normalize == "peak":
        processed = normalize_peak(trimmed, target_db=args.target_peak_db)
        fail_reason = "normalization_failed"
    else:  # "none"
        processed = trimmed
        fail_reason = None

    if processed is None:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        write_prompt_txt(rejected_dir, sample_id, raw_prompt)
        return {"status": "rejected", "reason": fail_reason}

    processed = apply_fade(processed, sr, fade_ms=args.fade_ms)

    out_path = out_dir / path.name
    with sf.SoundFile(
        str(out_path), mode="w",
        samplerate=sr, channels=processed.shape[1], subtype="PCM_24",
    ) as f:
        if raw_prompt:
            f.comment = raw_prompt    # RIFF INFO ICMT chunk
        f.software = SOFTWARE_TAG     # RIFF INFO ISFT chunk
        f.write(processed)

    write_prompt_txt(out_dir, sample_id, raw_prompt)
    return {"status": "ok"}


def resolve_paths(args):
    """Derive in/out/rejected paths when --category is given."""
    if args.category:
        outputs_dir = os.environ.get("SAS_OUTPUTS_DIR", "outputs")
        cat = args.category
        args.in_dir = args.in_dir or f"{outputs_dir}/raw/{cat}"
        args.out_dir = args.out_dir or f"{outputs_dir}/processed/{cat}"
        args.rejected_dir = args.rejected_dir or f"{outputs_dir}/rejected/{cat}"

    # Defaults if neither --category nor explicit dirs were given
    args.in_dir = args.in_dir or "outputs/raw"
    args.out_dir = args.out_dir or "outputs/processed"
    args.rejected_dir = args.rejected_dir or "outputs/rejected"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--category", default=None,
                        help="If set, derive --in-dir/--out-dir/--rejected-dir "
                             "from $SAS_OUTPUTS_DIR/{raw,processed,rejected}/<category>")
    parser.add_argument("--in-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--rejected-dir", default=None)
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=2.5)
    parser.add_argument("--trim-threshold-db", type=float, default=-45)
    parser.add_argument("--pad-ms", type=float, default=15)
    parser.add_argument("--fade-ms", type=float, default=5)
    parser.add_argument("--target-peak-db", type=float, default=-1.0,
                        help="Peak target when --normalize peak (legacy mode).")
    parser.add_argument("--normalize", choices=["lufs", "peak", "none"], default="lufs",
                        help="Loudness mode. lufs=BS.1770 perceived loudness "
                             "(default), peak=peak-only (legacy), none=skip gain stage.")
    parser.add_argument("--target-lufs", type=float, default=-16.0,
                        help="LUFS target when --normalize lufs. Default -16 LUFS.")
    parser.add_argument("--peak-ceiling-db", type=float, default=-1.0,
                        help="Hard peak ceiling applied AFTER LUFS gain (dBFS).")
    args = parser.parse_args()

    resolve_paths(args)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rejected_dir = Path(args.rejected_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        print(f"no WAVs found in {in_dir}; nothing to do")
        return

    results = []
    for wav in tqdm(wavs, desc=f"Post-processing {in_dir.name}"):
        results.append(process_file(wav, out_dir, rejected_dir, args))

    ok = sum(1 for r in results if r["status"] == "ok")
    rejected = len(results) - ok
    print(f"Processed: {ok}")
    print(f"Rejected: {rejected}")


if __name__ == "__main__":
    main()
