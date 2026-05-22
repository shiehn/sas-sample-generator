"""Wrap a flat list of pitched-instrument prompts into JSONL for batch_generate.py.

Sibling to list_to_jsonl.py (drums). Two structural differences from the
drum version:

  1. Per-prompt fanout: for categories with multiple target pitches
     (Phase 1.2 multi-source — Basses E1+E2, Pianos C2+C4, etc.), this
     emits ONE JSONL row per (prompt × target_pitch). Single-source
     categories (everything in Phase 1.0) get one row per prompt.

  2. ID hash incorporates target_pitch_midi: the drum hash is
     sha1(category:prompt:seed); the pitched hash adds the target pitch
     so two sources of the same prompt at different pitches don't
     collide. The hash stays content-addressed across re-runs, which is
     what `--skip-existing` in batch_generate.py relies on.

JSONL row shape consumed by batch_generate.py:

  {
    "id": "<category>-<hash8>",
    "category": "<category>",
    "prompt": "<positive prompt>",
    "negative_prompt": "<from pitched_category_config>",
    "seed": <int>,
    "duration": <float>,
    "target_pitch_midi": <int>          # pitched-only extension; ignored
                                        # by batch_generate but read by
                                        # gate_pitched + enrich_pitched
  }
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitched_category_config import PITCHED_CATEGORIES

CATEGORY_RE = re.compile(r"^[a-z0-9-]+$")


def short_hash(category: str, prompt: str, seed: int, target_pitch_midi: int) -> str:
    """First 8 hex chars of SHA1(category:prompt:seed:target_pitch). ~16M-key
    space. The target_pitch is part of the key so multi-source categories
    (same prompt at E1 + E2 etc.) don't collide on the filesystem."""
    payload = f"{category}:{prompt}:{seed}:{target_pitch_midi}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="in_path", required=True,
                        help="Path to text file (one prompt per line)")
    parser.add_argument("--out", dest="out_path", required=True,
                        help="Path to JSONL output file")
    parser.add_argument("--category", default=None,
                        help="Category name (default: input file stem). Must match ^[a-z0-9-]+$")
    parser.add_argument("--start-seed", type=int, default=2001,
                        help="Starting seed (default: 2001 — offset from drum's 1001 so the "
                             "two pipelines can't accidentally collide on the same prompt text)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        sys.exit(f"input not found: {in_path}")

    category = args.category or in_path.stem
    if not CATEGORY_RE.match(category):
        sys.exit(f"invalid category {category!r}: must match {CATEGORY_RE.pattern}")

    cfg = PITCHED_CATEGORIES.get(category)
    if cfg is None:
        sys.exit(
            f"unknown pitched category {category!r}. Add it to "
            f"scripts/pitched_category_config.py first."
        )

    seen: set[str] = set()
    rows: list[dict] = []
    seed_offset = 0

    for line_num, raw in enumerate(in_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        normalized = " ".join(line.lower().split())
        if normalized in seen:
            print(f"warning: line {line_num} duplicate, skipping: {line[:60]}", file=sys.stderr)
            continue
        seen.add(normalized)

        # Fan out across target pitches. Single-source categories produce one
        # row per prompt; multi-source (e.g. Basses E1+E2) produce two,
        # distinguished by the target_pitch in their hash.
        for target_pitch in cfg.target_pitches_midi:
            seed = args.start_seed + seed_offset
            rows.append({
                "id": f"{category}-{short_hash(category, line, seed, target_pitch)}",
                "category": category,
                "prompt": line,
                "negative_prompt": cfg.negative_prompt,
                "seed": seed,
                "duration": cfg.duration_seconds,
                "target_pitch_midi": target_pitch,
            })
            seed_offset += 1

    if not rows:
        sys.exit(f"no prompts found in {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(rows)} prompts to {out_path} "
        f"(category={category}, duration={cfg.duration_seconds}s, "
        f"target_pitches={list(cfg.target_pitches_midi)})"
    )


if __name__ == "__main__":
    main()
