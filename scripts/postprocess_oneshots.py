import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


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


def process_file(path, out_dir, rejected_dir, args):
    y, sr = sf.read(path, always_2d=True)

    trimmed, reject_reason = trim_silence(
        y,
        sr,
        threshold_db=args.trim_threshold_db,
        pad_ms=args.pad_ms,
    )

    if reject_reason:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        return {
            "file": path.name,
            "status": "rejected",
            "reason": reject_reason,
            "duration_seconds": 0,
            "output": str(rejected_path),
        }

    max_samples = int(args.max_seconds * sr)
    if len(trimmed) > max_samples:
        trimmed = trimmed[:max_samples]

    if args.mono:
        trimmed = np.mean(trimmed, axis=1, keepdims=True)

    processed = normalize_peak(trimmed, target_db=args.target_peak_db)
    if processed is None:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        return {
            "file": path.name,
            "status": "rejected",
            "reason": "normalization_failed",
            "duration_seconds": 0,
            "output": str(rejected_path),
        }

    processed = apply_fade(processed, sr, fade_ms=args.fade_ms)

    out_path = out_dir / path.name
    sf.write(out_path, processed, sr, subtype="PCM_24")

    return {
        "file": path.name,
        "status": "ok",
        "reason": "",
        "duration_seconds": round(len(processed) / sr, 4),
        "sample_rate": sr,
        "channels": processed.shape[1],
        "output": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="outputs/raw")
    parser.add_argument("--out-dir", default="outputs/processed")
    parser.add_argument("--rejected-dir", default="outputs/rejected")
    parser.add_argument("--manifest", default="outputs/manifests/processed_manifest.csv")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=2.5)
    parser.add_argument("--trim-threshold-db", type=float, default=-45)
    parser.add_argument("--pad-ms", type=float, default=15)
    parser.add_argument("--fade-ms", type=float, default=5)
    parser.add_argument("--target-peak-db", type=float, default=-1.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rejected_dir = Path(args.rejected_dir)
    manifest_path = Path(args.manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    rows = []

    for wav in tqdm(wavs, desc="Post-processing"):
        rows.append(process_file(wav, out_dir, rejected_dir, args))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    rejected = len(rows) - ok
    print(f"Processed: {ok}")
    print(f"Rejected: {rejected}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
