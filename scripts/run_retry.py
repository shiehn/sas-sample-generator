#!/usr/bin/env python3
"""Generate + gate with RETRY-TO-TARGET.

Each prompt should yield a surviving sample. A round:
  1. generate N candidates/prompt (batch_generate)
  2. gate them (gate_pitched / gate_drums), which keeps the best per prompt
  3. any prompt whose candidates ALL failed is regenerated next round with FRESH
     seeds (batch_generate --variant-offset) and ONLY those ids are re-gated.
Repeat until no prompt is left failing, or --max-retries rounds are spent (a hard
cap so a pathological prompt can't loop forever — remaining failures are logged).

Drives the existing scripts via subprocess; the pure helpers (failed_ids /
filter_rows) are unit-tested. Pipeline-agnostic (drums + pitched).

Usage (normally called by run_all.sh / run_pitched.sh):
  python scripts/run_retry.py --pipeline pitched --categories basses synths \\
      --outputs-dir outputs --steps 8 --batch-size 32 --max-retries 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def src_jsonl(pipeline: str, category: str) -> Path:
    sub = "prompts/pitched" if pipeline == "pitched" else "prompts"
    return REPO / sub / f"{category}.jsonl"


def gated_dir(pipeline: str, outputs_dir: Path, category: str) -> Path:
    sub = "gated" if pipeline == "pitched" else "gated_drums"
    return outputs_dir / sub / category


def failed_ids(gated_cat_dir: Path) -> set:
    """Base ids that ALL-failed the gate (one _failures/<id>.json each)."""
    fdir = gated_cat_dir / "_failures"
    return {p.stem for p in fdir.glob("*.json")} if fdir.is_dir() else set()


def filter_rows(src: Path, ids: set) -> list[str]:
    """Lines of `src` JSONL whose row id is in `ids` (preserves order)."""
    out: list[str] = []
    if not src.exists() or not ids:
        return out
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rid = json.loads(line).get("id")
        except json.JSONDecodeError:
            continue
        if rid in ids:
            out.append(line)
    return out


def _run(cmd: list[str]) -> None:
    print(f"[retry] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _generate(jsonls: list[Path], outputs_dir: Path, steps: int, batch: int,
              offset: int, anchor: bool) -> None:
    cmd = [sys.executable, str(REPO / "scripts/batch_generate.py"),
           "--prompts", *[str(j) for j in jsonls],
           "--out-root", str(outputs_dir / "raw"),
           "--steps", str(steps), "--batch-size", str(batch),
           "--variant-offset", str(offset), "--skip-existing"]
    if anchor:
        cmd.append("--init-audio-anchor")
    _run(cmd)


def _gate(pipeline: str, category: str, outputs_dir: Path, jsonl: Path,
          only_ids: Path | None) -> None:
    if pipeline == "pitched":
        # Passing the (possibly retry-filtered) jsonl restricts the gate to those
        # ids — others are skipped as orphans, so prior winners are left intact.
        _run([sys.executable, str(REPO / "scripts/gate_pitched.py"),
              "--category", category, "--jsonl", str(jsonl),
              "--outputs-dir", str(outputs_dir)])
    else:
        cmd = [sys.executable, str(REPO / "scripts/gate_drums.py"),
               "--category", category, "--outputs-dir", str(outputs_dir)]
        if only_ids is not None:
            cmd += ["--only-ids", str(only_ids)]
        _run(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True, choices=["pitched", "drums"])
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Extra rounds after round 0 (0 = no retry). Hard cap.")
    ap.add_argument("--stride", type=int, default=64,
                    help="Variant-index stride between rounds (must exceed the largest "
                         "per-category variant count so rounds never collide).")
    ap.add_argument("--init-audio-anchor", action="store_true")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    retry_dir = outputs_dir / "_retry" / args.pipeline
    retry_dir.mkdir(parents=True, exist_ok=True)

    # Round 0 processes every category from its full source jsonl.
    cur = {c: src_jsonl(args.pipeline, c) for c in args.categories
           if src_jsonl(args.pipeline, c).exists()}
    only_ids_for: dict[str, Path | None] = {c: None for c in cur}

    for r in range(args.max_retries + 1):
        if not cur:
            break
        offset = r * args.stride
        print(f"\n[retry] === round {r} (offset {offset}): {len(cur)} categories ===", flush=True)
        _generate(list(cur.values()), outputs_dir, args.steps, args.batch_size,
                  offset, args.init_audio_anchor)
        for cat, jl in cur.items():
            _gate(args.pipeline, cat, outputs_dir, jl, only_ids_for.get(cat))

        # Recompute failures across ALL categories; build next round's retry jsonls.
        nxt: dict[str, Path] = {}
        only_ids_for = {}
        total_fail = 0
        for cat in args.categories:
            fids = failed_ids(gated_dir(args.pipeline, outputs_dir, cat))
            rows = filter_rows(src_jsonl(args.pipeline, cat), fids)
            if not rows:
                continue
            total_fail += len(rows)
            rj = retry_dir / f"{cat}.jsonl"
            rj.write_text("\n".join(rows) + "\n", encoding="utf-8")
            ids_file = retry_dir / f"{cat}.ids"
            ids_file.write_text("\n".join(sorted(fids)) + "\n", encoding="utf-8")
            nxt[cat] = rj
            only_ids_for[cat] = ids_file

        if not nxt:
            print("[retry] all prompts satisfied — no failures remain.", flush=True)
            return
        if r == args.max_retries:
            print(f"[retry] max-retries reached; {total_fail} prompt(s) still failing "
                  f"across {len(nxt)} categories (left as _failures).", flush=True)
            return
        print(f"[retry] {total_fail} prompt(s) still failing → retrying.", flush=True)
        cur = nxt


if __name__ == "__main__":
    main()
