#!/usr/bin/env python3
"""Pitch-accuracy verification report for the pitched-instrument pipeline.

Read-only consumer of the gate stage's artefacts. For each category under
`<outputs>/gated/<cat>/` it reads the winner sidecars (`*.gate.json`) and the
fully-failed prompts (`_failures/*.json`) and emits, per category and overall:

  - pitch-class accuracy   round(measured) % 12 == target % 12   (the headline
    metric the spectral detector moved 15% -> 70%; v3 target is >= 80%)
  - exact-semitone accuracy round(measured) == target
  - median / p90 |cents_offset|  (measured-vs-target, includes semitone misses)
  - all-variants-fail rate and a per-rejection-reason tally
  - an ASCII histogram of signed cents_offset bucketed by semitone distance

It writes `<outputs>/_reports/pitch_summary.{json,md}` (the `_reports` dir is
`_`-prefixed so build_pack.py already excludes it from packs) so two runs diff
directly — this is the artefact the init_audio A/B (1d) and the pre-flight
acceptance gate are judged on.

Pure stdlib (json/pathlib/statistics) so it runs anywhere, no venv needed.

Usage:
  python scripts/pitch_report.py --outputs-dir outputs
  python scripts/pitch_report.py --categories basses synths    # subset
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Optional


# Signed-cents histogram bucket edges (cents). Centered so that ±50 == "on the
# requested semitone", ±150 == one semitone off, etc. — a readable view of how
# far SA3 landed from target.
_HIST_EDGES = [-250, -150, -50, 50, 150, 250]
_HIST_LABELS = [
    "<= -250", "-250..-150", "-150..-50", "-50..50 (on note)",
    "50..150", "150..250", "> 250",
]


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (pct in 0..100). None for empty input."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _hist_bucket(cents: float) -> int:
    """Index into _HIST_LABELS for a signed cents value."""
    for i, edge in enumerate(_HIST_EDGES):
        if cents <= edge:
            return i
    return len(_HIST_EDGES)


def _ascii_bar(count: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return ""
    filled = int(round(width * count / total))
    return "#" * filled


def analyze_category(cat: str, gated_dir: Path) -> Optional[dict]:
    """Aggregate one category's gate artefacts into a stats dict, or None if
    the category has no gated output yet."""
    winner_files = sorted(gated_dir.glob("*.gate.json"))
    failures_dir = gated_dir / "_failures"
    failure_files = sorted(failures_dir.glob("*.json")) if failures_dir.is_dir() else []

    if not winner_files and not failure_files:
        return None

    cents_signed: list[float] = []
    abs_cents: list[float] = []
    confidences: list[float] = []
    pitch_class_hits = 0
    exact_semitone_hits = 0
    pitched_winners = 0          # winners with a finite measured pitch
    reason_tally: Counter = Counter()
    hist = [0] * len(_HIST_LABELS)

    for wf in winner_files:
        try:
            data = json.loads(wf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        target = data.get("target_pitch_midi")
        winner = data.get("winner") or {}
        metrics = winner.get("metrics") or {}
        measured = metrics.get("measured_pitch_midi")
        cents = metrics.get("measured_pitch_cents_offset")
        conf = metrics.get("pitch_confidence")

        # Per-variant rejection reasons (rejected siblings of the winner).
        for v in data.get("all_variants") or []:
            reason = v.get("rejection_reason")
            if reason:
                reason_tally[reason] += 1

        if conf is not None:
            confidences.append(float(conf))

        if measured is None or target is None:
            continue  # unpitched (e.g. fx) — counted as a winner but not in pitch stats
        pitched_winners += 1
        if round(measured) % 12 == int(target) % 12:
            pitch_class_hits += 1
        if round(measured) == int(target):
            exact_semitone_hits += 1
        if cents is not None and math.isfinite(float(cents)):
            cents_signed.append(float(cents))
            abs_cents.append(abs(float(cents)))
            hist[_hist_bucket(float(cents))] += 1

    for ff in failure_files:
        try:
            data = json.loads(ff.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for reason, n in (data.get("rejection_reasons") or {}).items():
            reason_tally[reason] += int(n)

    n_winners = len(winner_files)
    n_failures = len(failure_files)
    total_prompts = n_winners + n_failures

    return {
        "category": cat,
        "total_prompts": total_prompts,
        "winners": n_winners,
        "failures": n_failures,
        "survival_rate": (n_winners / total_prompts) if total_prompts else 0.0,
        "fail_rate": (n_failures / total_prompts) if total_prompts else 0.0,
        "pitched_winners": pitched_winners,
        "pitch_class_accuracy": (pitch_class_hits / pitched_winners) if pitched_winners else None,
        "exact_semitone_accuracy": (exact_semitone_hits / pitched_winners) if pitched_winners else None,
        "median_abs_cents": _percentile(abs_cents, 50),
        "p90_abs_cents": _percentile(abs_cents, 90),
        "median_confidence": _percentile(confidences, 50),
        "cents_histogram": dict(zip(_HIST_LABELS, hist)),
        "rejection_reasons": dict(reason_tally.most_common()),
    }


def _fmt_pct(x: Optional[float]) -> str:
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def _fmt_cents(x: Optional[float]) -> str:
    return f"{x:.1f}" if x is not None else "n/a"


def build_markdown(cats: list[dict], overall: dict) -> str:
    lines: list[str] = []
    lines.append("# Pitch-accuracy report")
    lines.append("")
    lines.append(
        "Measured at the **gate** stage (how close SA3 landed to the requested "
        "target pitch, before enrich snaps each sample to an integer semitone)."
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- instruments (winners): **{overall['winners']}** of {overall['total_prompts']} prompts "
                 f"(survival {_fmt_pct(overall['survival_rate'])})")
    lines.append(f"- **pitch-class accuracy: {_fmt_pct(overall['pitch_class_accuracy'])}** "
                 f"(v3 target >= 80%)")
    lines.append(f"- exact-semitone accuracy: {_fmt_pct(overall['exact_semitone_accuracy'])}")
    lines.append(f"- |cents| median / p90: {_fmt_cents(overall['median_abs_cents'])} / "
                 f"{_fmt_cents(overall['p90_abs_cents'])}")
    lines.append("")
    lines.append("## Per-category")
    lines.append("")
    lines.append("| category | winners | survival | pitch-class | exact-semi | med\\|¢\\| | p90\\|¢\\| |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for c in cats:
        lines.append(
            f"| {c['category']} | {c['winners']} | {_fmt_pct(c['survival_rate'])} | "
            f"{_fmt_pct(c['pitch_class_accuracy'])} | {_fmt_pct(c['exact_semitone_accuracy'])} | "
            f"{_fmt_cents(c['median_abs_cents'])} | {_fmt_cents(c['p90_abs_cents'])} |"
        )
    lines.append("")
    for c in cats:
        lines.append(f"### {c['category']}")
        lines.append("")
        hist = c["cents_histogram"]
        hist_total = sum(hist.values())
        if hist_total:
            lines.append("signed cents offset (measured − target):")
            lines.append("```")
            for label in _HIST_LABELS:
                n = hist[label]
                lines.append(f"{label:>18} | {_ascii_bar(n, hist_total)} {n}")
            lines.append("```")
        if c["rejection_reasons"]:
            top = list(c["rejection_reasons"].items())[:8]
            lines.append("rejections (per variant): "
                         + ", ".join(f"{r}={n}" for r, n in top))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default=None,
                        help="Root holding gated/ (default: $SAS_OUTPUTS_DIR or outputs)")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Subset of categories. Default: every dir under gated/.")
    parser.add_argument("--out", default=None,
                        help="Report dir (default: <outputs>/_reports)")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir or os.environ.get("SAS_OUTPUTS_DIR", "outputs"))
    gated_root = outputs_dir / "gated"
    if not gated_root.is_dir():
        raise SystemExit(f"[pitch_report] no gated/ dir under {outputs_dir} — run the gate stage first")

    if args.categories:
        cat_names = list(args.categories)
    else:
        cat_names = sorted(p.name for p in gated_root.iterdir()
                           if p.is_dir() and not p.name.startswith("_"))

    cats: list[dict] = []
    for cat in cat_names:
        stats = analyze_category(cat, gated_root / cat)
        if stats is not None:
            cats.append(stats)

    if not cats:
        raise SystemExit(f"[pitch_report] no gate artefacts found under {gated_root}")

    # Overall = winner-weighted aggregate across categories.
    tot_winners = sum(c["winners"] for c in cats)
    tot_prompts = sum(c["total_prompts"] for c in cats)
    tot_pitched = sum(c["pitched_winners"] for c in cats)
    pc_hits = sum((c["pitch_class_accuracy"] or 0) * c["pitched_winners"] for c in cats)
    ex_hits = sum((c["exact_semitone_accuracy"] or 0) * c["pitched_winners"] for c in cats)
    all_abs_meds = [c["median_abs_cents"] for c in cats if c["median_abs_cents"] is not None]
    all_abs_p90 = [c["p90_abs_cents"] for c in cats if c["p90_abs_cents"] is not None]
    overall = {
        "winners": tot_winners,
        "total_prompts": tot_prompts,
        "survival_rate": (tot_winners / tot_prompts) if tot_prompts else 0.0,
        "pitch_class_accuracy": (pc_hits / tot_pitched) if tot_pitched else None,
        "exact_semitone_accuracy": (ex_hits / tot_pitched) if tot_pitched else None,
        # overall |cents| reported as the median of per-category medians (robust,
        # avoids re-reading every file) — fine for run-to-run comparison.
        "median_abs_cents": _percentile(all_abs_meds, 50),
        "p90_abs_cents": _percentile(all_abs_p90, 90),
    }

    out_dir = Path(args.out) if args.out else outputs_dir / "_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"overall": overall, "categories": cats}
    (out_dir / "pitch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "pitch_summary.md").write_text(build_markdown(cats, overall), encoding="utf-8")

    # Console digest.
    print(f"[pitch_report] {len(cats)} categories, {tot_winners} instruments")
    print(f"[pitch_report] OVERALL pitch-class accuracy: {_fmt_pct(overall['pitch_class_accuracy'])} "
          f"(target >= 80%)")
    for c in cats:
        print(f"  {c['category']:<16} pc={_fmt_pct(c['pitch_class_accuracy']):>6} "
              f"survival={_fmt_pct(c['survival_rate']):>6} med|c|={_fmt_cents(c['median_abs_cents'])}")
    print(f"[pitch_report] wrote {out_dir / 'pitch_summary.md'}")


if __name__ == "__main__":
    main()
