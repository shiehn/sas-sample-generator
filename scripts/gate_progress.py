#!/usr/bin/env python3
"""Read-only progress meter for an in-flight pitched (or drum) gate run.

Counts gated winners vs failures per category and — crucially — tells FRESH
failures (real rejection reasons from the current gate) apart from STALE ones
left over from an earlier/aborted run, whose `rejection_reasons` still contain
`read_failed`. Stale failures are jobs the current re-gate HASN'T REACHED yet,
so they count as PENDING, not processed.

Safe to run anytime from a second SSH session — it only reads files, never
touches the running gate.

  python3 scripts/gate_progress.py
  python3 scripts/gate_progress.py --outputs-dir /workspace/outputs
  python3 scripts/gate_progress.py --gated-subdir gated_drums      # drum gate
"""
import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir",
                    default=os.environ.get("SAS_OUTPUTS_DIR", "/workspace/outputs"))
    ap.add_argument("--gated-subdir", default="gated",
                    help="'gated' (pitched, default) or 'gated_drums'")
    args = ap.parse_args()

    root = Path(args.outputs_dir) / args.gated_subdir
    if not root.is_dir():
        print(f"[gate_progress] no such dir: {root}")
        return

    cats = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    tot_w = tot_fresh = tot_stale = 0
    cats_done = 0
    rows = []
    for d in cats:
        winners = sum(1 for _ in d.glob("*.gate.json"))
        fails = list((d / "_failures").glob("*.json"))
        stale = 0
        for f in fails:
            try:
                if "read_failed" in f.read_text(errors="ignore"):
                    stale += 1
            except OSError:
                pass
        fresh = len(fails) - stale
        tot_w += winners
        tot_fresh += fresh
        tot_stale += stale
        done = (stale == 0)
        if done:
            cats_done += 1
        status = "PENDING" if (stale and winners + fresh == 0) else ("partial" if stale else "done")
        rows.append((d.name, winners, fresh, stale, status))

    for name, w, fr, st, status in rows:
        print(f"  {name:16} pass={w:<5} fail={fr:<5} pending={st:<5} {status}")

    processed = tot_w + tot_fresh
    total = processed + tot_stale
    pct = (100.0 * processed / total) if total else 0.0
    print()
    print(f"[gate_progress] categories done: {cats_done}/{len(cats)}")
    print(f"[gate_progress] jobs gated:      {processed:,}/{total:,} = {pct:.1f}% "
          f"({tot_stale:,} pending, {tot_w:,} winners so far)")


if __name__ == "__main__":
    main()
