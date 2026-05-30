#!/usr/bin/env python3
"""Gate pitched categories IN PARALLEL — one gate_pitched.py per category, N at a
time — so an idle multi-core pod isn't bottlenecked by the single-threaded serial
run_pitched loop. gate_pitched is independent per category, so running them
concurrently gives roughly an N-way speedup.

RESUMES safely: by default it SKIPS categories that are already fully gated (have
winners and no leftover `read_failed` failures), so it picks up only the work the
serial run hadn't reached — nothing finished is redone.

Each worker pins its libs to 1 thread (OMP/MKL/TF), so N workers use N cores
cleanly instead of N×M threads thrashing the box.

  python3 scripts/gate_parallel.py --dry-run        # show what it WOULD gate, run nothing
  python3 scripts/gate_parallel.py                  # gate all not-yet-done categories
  python3 scripts/gate_parallel.py --workers 12     # cap concurrency (default: cpu-2)
  python3 scripts/gate_parallel.py --categories basses strings   # explicit subset
  python3 scripts/gate_parallel.py --all            # re-gate every category from scratch

Stop the old serial gate first (its finished categories are saved). Monitor with
scripts/gate_progress.py from another shell.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Pin each worker's math libs to one thread so N workers == N cores (no thrash).
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1",
}


def enabled_categories() -> list:
    f = REPO / "scripts" / "pitched_categories.txt"
    cats = []
    for ln in f.read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            cats.append(ln)
    return cats


def needs_gating(outputs_dir: Path, cat: str) -> bool:
    """True if the category still has work. A leftover `read_failed` failure means
    a job was never freshly gated (stale from an aborted run). A category with
    winners and no stale failures is done -> skip. Never-gated -> gate."""
    gdir = outputs_dir / "gated" / cat
    fdir = gdir / "_failures"
    has_winner = any(gdir.glob("*.gate.json")) if gdir.is_dir() else False
    if fdir.is_dir():
        for f in fdir.glob("*.json"):
            try:
                if "read_failed" in f.read_text(errors="ignore"):
                    return True  # stale -> not yet freshly gated
            except OSError:
                pass
    return not has_winner


def gate_one(cat: str, outputs_dir: Path):
    jsonl = REPO / "prompts" / "pitched" / f"{cat}.jsonl"
    if not jsonl.exists():
        return cat, 1, f"missing jsonl: {jsonl}"
    cmd = [sys.executable, str(REPO / "scripts/gate_pitched.py"),
           "--category", cat, "--jsonl", str(jsonl), "--outputs-dir", str(outputs_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, **_SINGLE_THREAD_ENV})
    summary = ""
    if r.stdout:
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        summary = lines[-1] if lines else ""
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        summary = (summary + " | " + (tail[-1] if tail else "no stderr"))[-300:]
    return cat, r.returncode, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=os.environ.get("SAS_OUTPUTS_DIR", "/workspace/outputs"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Explicit category list (default: all enabled).")
    ap.add_argument("--all", action="store_true",
                    help="Gate every category even if already done.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan, run nothing.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    base = args.categories if args.categories else enabled_categories()
    if args.all or args.categories:
        todo, skipped = list(base), []
    else:
        todo = [c for c in base if needs_gating(outputs_dir, c)]
        skipped = [c for c in base if c not in todo]

    workers = min(args.workers, max(1, len(todo)))
    print(f"[gate_parallel] workers={workers}  outputs={outputs_dir}")
    if skipped:
        print(f"[gate_parallel] skip {len(skipped)} already-done: {' '.join(skipped)}")
    print(f"[gate_parallel] gate {len(todo)}: {' '.join(todo) or '(none)'}")
    if args.dry_run:
        print("[gate_parallel] --dry-run: nothing executed.")
        return
    if not todo:
        print("[gate_parallel] nothing to do.")
        return

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(gate_one, c, outputs_dir): c for c in todo}
        for fut in as_completed(futs):
            cat, rc, summary = fut.result()
            done += 1
            print(f"[gate_parallel] ({done}/{len(todo)}) {'OK ' if rc == 0 else 'ERR'} {cat}: {summary}",
                  flush=True)
    print("[gate_parallel] all done. Run gate_progress.py to confirm, then STAGES=enrich.")


if __name__ == "__main__":
    main()
