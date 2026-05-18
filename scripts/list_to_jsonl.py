"""Wrap a flat list of prompt descriptions into JSONL for batch_generate.py.

Input: a UTF-8 text file with one description per line. Blank lines and
lines starting with '#' are ignored. Duplicate lines are skipped with a
warning.

Output: JSONL with one object per line, shaped:
  {"id": "<prefix>_NNNN", "prompt": "<line>", "seed": <int>, "duration": <float>}

The ID prefix defaults to the input file's stem (e.g. kicks.txt -> kicks_0001).
Override with --prefix if you want singular IDs (e.g. --prefix kick).
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", required=True, help="Path to text file (one description per line)")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to JSONL output file")
    parser.add_argument("--prefix", default=None, help="ID prefix (default: input file stem)")
    parser.add_argument("--start-seed", type=int, default=1001, help="Starting seed (default: 1001)")
    parser.add_argument("--start-id", type=int, default=1, help="Starting ID number (default: 1)")
    parser.add_argument("--duration", type=float, default=1.5, help="Duration in seconds for every row (default: 1.5)")
    parser.add_argument("--pad", type=int, default=4, help="Zero-pad width for IDs (default: 4)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    prefix = args.prefix or in_path.stem

    seen: set[str] = set()
    rows: list[dict] = []

    for line_num, raw in enumerate(in_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        normalized = " ".join(line.lower().split())
        if normalized in seen:
            print(f"warning: line {line_num} duplicate, skipping: {line[:60]}", file=sys.stderr)
            continue
        seen.add(normalized)

        idx = args.start_id + len(rows)
        rows.append({
            "id": f"{prefix}_{idx:0{args.pad}d}",
            "prompt": line,
            "seed": args.start_seed + len(rows),
            "duration": args.duration,
        })

    if not rows:
        sys.exit(f"no prompts found in {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} prompts to {out_path}")


if __name__ == "__main__":
    main()
