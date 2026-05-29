#!/usr/bin/env python3
"""Generate + gate with a per-category MINIMUM survivor target.

Goal: every category ends with at least --target surviving samples (default 0 =
"no minimum, just retry outright failures"). A surviving sample = a prompt that
yields at least one gated winner (for multi-source pitched, a prompt counts once
no matter how many of its pitches pass — that's one merged instrument).

Each round generates candidates, gates them, then per category escalates while
still short of target:
  1. RE-ROLL failed prompts with FRESH seeds (--variant-offset), re-gate just
     those ids — recovers prompts that failed by bad luck. Up to --max-retries.
  2. TOP-UP: if re-rolls are exhausted and still short, author MORE prompts
     (gen_prompts.py preserves existing + appends), rebuild the JSONL, and
     generate + gate the new prompts. Up to --max-topups.
A hard round cap (max-retries + max-topups + 1) guarantees termination; any
remaining shortfall is logged, never looped on forever.

Model load is amortized: one batch_generate call per round covers ALL categories
still needing work (don't reload the 1.4B model per category).

Pure helpers (prompt_map / gated_ids / assess) are unit-tested; generate / gate /
author shell out to the existing scripts.
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


def prompt_txt(pipeline: str, category: str) -> Path:
    sub = "prompts/pitched" if pipeline == "pitched" else "prompts"
    return REPO / sub / f"{category}.txt"


def gated_dir(pipeline: str, outputs_dir: Path, category: str) -> Path:
    sub = "gated" if pipeline == "pitched" else "gated_drums"
    return outputs_dir / sub / category


# ----------------------------- pure helpers (unit-tested) -----------------------------

def prompt_map(src: Path) -> dict[str, str]:
    """id -> prompt for every row of a JSONL."""
    out: dict[str, str] = {}
    if not src.exists():
        return out
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[row["id"]] = row.get("prompt", row["id"])
    return out


def gated_ids(pipeline: str, outputs_dir: Path, category: str) -> set:
    """Base ids that produced a gated winner."""
    d = gated_dir(pipeline, outputs_dir, category)
    if not d.is_dir():
        return set()
    if pipeline == "pitched":
        return {p.name[: -len(".gate.json")] for p in d.glob("*.gate.json")}
    return {p.stem for p in d.glob("*.wav")}


def assess(pipeline: str, outputs_dir: Path, category: str, src: Path):
    """Return (survivors, failed_rows, failed_ids, pool_prompts).

    survivors      = distinct PROMPTS with >=1 gated winner (== instruments for
                     multi-source pitched; == samples for drums).
    failed_rows    = JSONL lines whose prompt has NO surviving id (to re-roll).
    failed_ids     = the ids of those rows (for gate --only-ids).
    pool_prompts   = distinct prompts in the source pool.
    """
    pof = prompt_map(src)
    all_prompts = set(pof.values())
    gids = gated_ids(pipeline, outputs_dir, category)
    surviving_prompts = {pof[i] for i in gids if i in pof}
    failed_prompts = all_prompts - surviving_prompts

    failed_rows: list[str] = []
    failed_ids: set = set()
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pof.get(row["id"]) in failed_prompts:
                failed_rows.append(line)
                failed_ids.add(row["id"])
    return len(surviving_prompts), failed_rows, failed_ids, len(all_prompts)


def rows_with_new_ids(src: Path, seen_ids: set) -> tuple[list[str], set]:
    """JSONL lines whose id is NOT already in seen_ids (the freshly-authored
    top-up prompts), plus their id set."""
    rows: list[str] = []
    ids: set = set()
    if not src.exists():
        return rows, ids
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rid = json.loads(line)["id"]
        except (json.JSONDecodeError, KeyError):
            continue
        if rid not in seen_ids:
            rows.append(line)
            ids.add(rid)
    return rows, ids


def ids_in(jsonl: Path) -> set:
    s: set = set()
    if not jsonl.exists():
        return s
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            s.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return s


# ----------------------------- subprocess drivers -----------------------------

def _run(cmd: list[str]) -> None:
    print(f"[retry] $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _generate(jsonls, outputs_dir: Path, steps: int, batch: int, offset: int, anchor: bool) -> None:
    if not jsonls:
        return
    cmd = [sys.executable, str(REPO / "scripts/batch_generate.py"),
           "--prompts", *[str(j) for j in jsonls],
           "--out-root", str(outputs_dir / "raw"),
           "--steps", str(steps), "--batch-size", str(batch),
           "--variant-offset", str(offset), "--skip-existing"]
    if anchor:
        cmd.append("--init-audio-anchor")
    _run(cmd)


def _gate(pipeline: str, category: str, outputs_dir: Path, jsonl: Path, only_ids: Path | None) -> None:
    if pipeline == "pitched":
        # The jsonl passed restricts the gate to those ids (others are orphans).
        _run([sys.executable, str(REPO / "scripts/gate_pitched.py"),
              "--category", category, "--jsonl", str(jsonl), "--outputs-dir", str(outputs_dir)])
    else:
        cmd = [sys.executable, str(REPO / "scripts/gate_drums.py"),
               "--category", category, "--outputs-dir", str(outputs_dir)]
        if only_ids is not None:
            cmd += ["--only-ids", str(only_ids)]
        _run(cmd)


def _author_more(pipeline: str, category: str, new_total: int) -> int:
    """Append prompts to the category's .txt (gen_prompts preserves existing) and
    rebuild its JSONL. Returns how many prompts the pool grew by (0 = tapped out)."""
    txt = prompt_txt(pipeline, category)
    before = _count_prompts(txt)
    _run([sys.executable, str(REPO / "scripts/gen_prompts.py"),
          "--only", category, "--target", str(new_total)])
    # Rebuild the JSONL from the now-larger prompt file.
    if pipeline == "pitched":
        _run([sys.executable, str(REPO / "scripts/list_to_jsonl_pitched.py"),
              "--in", str(txt), "--out", str(src_jsonl(pipeline, category))])
    else:
        _run([sys.executable, str(REPO / "scripts/list_to_jsonl.py"),
              "--in", str(txt), "--out", str(src_jsonl(pipeline, category))])
    return _count_prompts(txt) - before


def _count_prompts(txt: Path) -> int:
    if not txt.exists():
        return 0
    return sum(1 for ln in txt.read_text(encoding="utf-8").splitlines()
               if ln.strip() and not ln.strip().startswith("#"))


# ----------------------------------- main loop -----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", required=True, choices=["pitched", "drums"])
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--outputs-dir", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--target", type=int, default=0,
                    help="Minimum surviving samples per category. 0 = no minimum "
                         "(only re-roll outright failures).")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Re-roll rounds for failed prompts before switching to top-up.")
    ap.add_argument("--max-topups", type=int, default=3,
                    help="Prompt-authoring rounds when the pool can't reach --target.")
    ap.add_argument("--topup-margin", type=int, default=25,
                    help="Extra prompts to author beyond the bare shortfall (gating attrition).")
    ap.add_argument("--stride", type=int, default=64,
                    help="Variant-index stride between rounds (must exceed the largest "
                         "per-category variant count so rounds never collide).")
    ap.add_argument("--init-audio-anchor", action="store_true")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    retry_dir = outputs_dir / "_retry" / args.pipeline
    retry_dir.mkdir(parents=True, exist_ok=True)

    cats = [c for c in args.categories if src_jsonl(args.pipeline, c).exists()]
    state = {c: {"retries": 0, "topups": 0, "seen": set()} for c in cats}

    # Round 0: every category, from its full source pool.
    cur = {c: src_jsonl(args.pipeline, c) for c in cats}
    only_ids = {c: None for c in cats}

    hard_cap = args.max_retries + args.max_topups + 1
    for rnd in range(hard_cap + 1):
        if not cur:
            break
        offset = rnd * args.stride
        print(f"\n[retry] === round {rnd} (offset {offset}): {len(cur)} categor(ies) ===", flush=True)
        _generate(list(cur.values()), outputs_dir, args.steps, args.batch_size, offset, args.init_audio_anchor)
        for c, jl in cur.items():
            _gate(args.pipeline, c, outputs_dir, jl, only_ids.get(c))
            state[c]["seen"] |= ids_in(jl)

        # Decide the next round's work per category.
        nxt: dict[str, Path] = {}
        only_ids = {}
        for c in cats:
            src = src_jsonl(args.pipeline, c)
            surv, failed_rows, failed_ids, pool = assess(args.pipeline, outputs_dir, c, src)
            st = state[c]

            # No minimum: just clear outright failures, then stop.
            if args.target <= 0:
                if failed_rows and st["retries"] < args.max_retries:
                    st["retries"] += 1
                    rj = retry_dir / f"{c}.jsonl"; rj.write_text("\n".join(failed_rows) + "\n")
                    idf = retry_dir / f"{c}.ids"; idf.write_text("\n".join(sorted(failed_ids)) + "\n")
                    nxt[c] = rj; only_ids[c] = idf
                continue

            if surv >= args.target:
                print(f"[retry] {c}: {surv}/{args.target} ✓", flush=True)
                continue

            # Still short of target.
            if failed_rows and st["retries"] < args.max_retries:
                st["retries"] += 1
                print(f"[retry] {c}: {surv}/{args.target} — re-roll {len(failed_ids)} failed "
                      f"(retry {st['retries']}/{args.max_retries})", flush=True)
                rj = retry_dir / f"{c}.jsonl"; rj.write_text("\n".join(failed_rows) + "\n")
                idf = retry_dir / f"{c}.ids"; idf.write_text("\n".join(sorted(failed_ids)) + "\n")
                nxt[c] = rj; only_ids[c] = idf
            elif st["topups"] < args.max_topups:
                st["topups"] += 1
                new_total = pool + (args.target - surv) + args.topup_margin
                print(f"[retry] {c}: {surv}/{args.target} — re-rolls spent; authoring more "
                      f"prompts -> {new_total} (top-up {st['topups']}/{args.max_topups})", flush=True)
                grew = _author_more(args.pipeline, c, new_total)
                if grew <= 0:
                    print(f"[retry] {c}: prompt pool tapped out (combinatorial space); "
                          f"stuck at {surv}/{args.target}.", flush=True)
                    continue
                new_rows, new_ids = rows_with_new_ids(src, st["seen"])
                if not new_rows:
                    print(f"[retry] {c}: no new prompt rows; stuck at {surv}/{args.target}.", flush=True)
                    continue
                st["retries"] = 0  # the new prompts get their own re-roll budget
                rj = retry_dir / f"{c}.jsonl"; rj.write_text("\n".join(new_rows) + "\n")
                idf = retry_dir / f"{c}.ids"; idf.write_text("\n".join(sorted(new_ids)) + "\n")
                nxt[c] = rj; only_ids[c] = idf
            else:
                print(f"[retry] {c}: exhausted retries + top-ups; stuck at {surv}/{args.target}.", flush=True)

        cur = nxt

    # Final report.
    print("\n[retry] ===== final per-category survivor counts =====", flush=True)
    short = 0
    for c in cats:
        surv, _, _, _ = assess(args.pipeline, outputs_dir, c, src_jsonl(args.pipeline, c))
        mark = "✓" if (args.target <= 0 or surv >= args.target) else "SHORT"
        if mark == "SHORT":
            short += 1
        print(f"[retry]   {c:16} {surv}" + (f"/{args.target} {mark}" if args.target > 0 else ""), flush=True)
    if short:
        print(f"[retry] WARNING: {short} categor(ies) below target {args.target} "
              f"(raise --max-topups or check gate reject reasons).", flush=True)


if __name__ == "__main__":
    main()
