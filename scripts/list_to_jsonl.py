"""Wrap a flat list of prompt descriptions into JSONL for batch_generate.py.

Input: a UTF-8 text file with one description per line. Blank lines and
lines starting with '#' are ignored. Duplicate lines are skipped with a
warning. The filename stem is used as the category by default
(override with --category).

Output JSONL row shape:
  {
    "id": "<category>-<hash8>",
    "category": "<category>",
    "prompt": "<line>",
    "negative_prompt": "<from category_config>",
    "seed": <int>,
    "duration": <float>
  }

`id` is content-addressed: same (category, prompt, seed) -> same hash,
which keeps `batch_generate.py --skip-existing` honest across re-runs.
Editing a prompt line changes the hash, so the corresponding WAV becomes
orphaned in outputs/raw/<category>/ — that's expected; delete it if you
care.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Make sibling-module import work no matter where this is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from category_config import (
    CATEGORY_DURATIONS,
    CATEGORY_NEGATIVES,
    DEFAULT_NEGATIVE_PROMPT,
)

CATEGORY_RE = re.compile(r"^[a-z0-9-]+$")


def short_hash(category: str, prompt: str, seed: int) -> str:
    """First 8 hex chars of SHA1(category:prompt:seed). ~16M-key space."""
    h = hashlib.sha1(f"{category}:{prompt}:{seed}".encode("utf-8")).hexdigest()
    return h[:8]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="Path to text file (one description per line)")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="Path to JSONL output file")
    parser.add_argument("--category", default=None,
                        help="Category name (default: input file stem). Must match ^[a-z0-9-]+$")
    parser.add_argument("--start-seed", type=int, default=1001,
                        help="Starting seed (default: 1001)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override per-category default duration")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    category = args.category or in_path.stem
    if not CATEGORY_RE.match(category):
        sys.exit(f"invalid category {category!r}: must match {CATEGORY_RE.pattern}")

    negative_prompt = CATEGORY_NEGATIVES.get(category, DEFAULT_NEGATIVE_PROMPT)
    duration = (
        args.duration if args.duration is not None
        else CATEGORY_DURATIONS.get(category, 1.5)
    )

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

        seed = args.start_seed + len(rows)
        rows.append({
            "id": f"{category}-{short_hash(category, line, seed)}",
            "category": category,
            "prompt": line,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "duration": duration,
        })

    if not rows:
        sys.exit(f"no prompts found in {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(rows)} prompts to {out_path} "
        f"(category={category}, duration={duration}s)"
    )


if __name__ == "__main__":
    main()
