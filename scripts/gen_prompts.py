#!/usr/bin/env python3
"""Combinatorial prompt generator for the v3 sample run.

Produces ~N (default 200) diverse, deduped prompts per category in the existing
house style:

    <style> <tone> <noun> one shot, <character>, <decay>, dry, <exclusions>, no loop

Design goals:
  - EDM/electronic bias (~58% EDM / ~25% urban / ~17% acoustic-orchestral-world),
    overridable per category.
  - PRESERVE existing curated prompts verbatim (keeps their content-addressed
    hashes stable, so already-generated WAVs aren't orphaned) — we only ADD new
    lines on top until the target count is reached.
  - Near-duplicate dedup (token-set Jaccard) on top of the exact dedup the
    list_to_jsonl scripts already do.
  - Deterministic: seeded per category, so re-runs are byte-stable. (Finalize
    wording before the GPU run — edits change hashes.)

Usage:
  python scripts/gen_prompts.py                 # all categories, target 200
  python scripts/gen_prompts.py --target 200 --only kick 808 banjos
  python scripts/gen_prompts.py --dry-run       # report counts, write nothing
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------

EDM_STYLES = [
    "deep house", "tech house", "techno", "acid techno", "minimal techno",
    "trance", "big room", "future bass", "dubstep", "drum and bass", "trap edm",
    "hardstyle", "synthwave", "electro", "uk garage", "2-step", "breakbeat",
    "progressive house", "melodic techno", "glitch", "ambient techno", "phonk",
]
URBAN_STYLES = [
    "hip-hop", "lo-fi hip-hop", "boom bap", "trap", "drill", "r&b", "neo-soul",
    "cloud rap", "west coast", "memphis",
]
ACOUSTIC_STYLES = [
    "orchestral", "cinematic", "jazz", "folk", "world", "classical", "chamber",
    "ambient", "soundtrack",
]
STYLE_POOLS = {"edm": EDM_STYLES, "urban": URBAN_STYLES, "acoustic": ACOUSTIC_STYLES}
DEFAULT_WEIGHTS = {"edm": 0.58, "urban": 0.25, "acoustic": 0.17}
WEIGHT_PRESETS = {
    "edm_heavy": {"edm": 0.8, "urban": 0.15, "acoustic": 0.05},
    "urban": {"edm": 0.3, "urban": 0.55, "acoustic": 0.15},
    "acoustic": {"edm": 0.15, "urban": 0.2, "acoustic": 0.65},
    "balanced": DEFAULT_WEIGHTS,
}

TONE = [
    "warm", "bright", "dark", "gritty", "punchy", "lush", "metallic", "woody",
    "airy", "fat", "detuned", "clean", "vintage", "glassy", "hollow", "crisp",
    "mellow", "aggressive", "smooth", "dusty", "saturated", "wide",
]
PROCESSING = [
    "dry", "dry", "dry", "saturated", "lo-fi", "vinyl-warmed", "tape-saturated",
    "analog", "filtered", "compressed", "clean studio sample", "mono-compatible",
]

# Per-family character + decay word lists (the two descriptive clauses).
FAMILY = {
    "kick":      (["hard click transient", "punchy body", "tight attack", "soft transient", "sharp click"],
                  ["short punchy decay", "controlled low end", "deep sub tail", "snappy decay", "round low end"]),
    "snare":     (["sharp crack", "tight body", "snappy attack", "fat body", "crisp snap"],
                  ["short decay", "controlled tail", "punchy body", "quick decay"]),
    "hat":       (["crisp transient", "tight metallic tick", "sharp top end", "bright shimmer"],
                  ["short decay", "quick close", "snappy tail", "tight decay"]),
    "cymbal":    (["bright shimmer", "metallic wash", "sharp attack", "splashy transient"],
                  ["long ringing tail", "decaying wash", "lingering shimmer", "smooth decay"]),
    "perc":      (["crisp transient", "snappy attack", "dry tick", "bright snap"],
                  ["short decay", "tight tail", "quick decay", "natural decay"]),
    "tom":       (["round attack", "punchy body", "deep thud", "warm hit"],
                  ["medium decay", "controlled tail", "resonant body", "short decay"]),
    "clap":      (["tight snappy transient", "layered claps", "sharp single clap", "wide stack"],
                  ["short decay", "dry snap", "punchy tail", "controlled decay"]),
    "sub":       (["pure sine sub", "deep tuned low end", "gritty saturated sub", "tight controlled sub"],
                  ["long smooth decay", "slow sub tail", "boomy sustain", "fast decay"]),
    "riser":     (["sweeping pitch build-up", "rising filtered noise", "increasing intensity", "tension build"],
                  ["long upward sweep", "pre-drop tension", "accelerating rise", "smooth crescendo"]),
    "downer":    (["descending pitch fall", "filtered noise drop", "releasing energy", "downward motion"],
                  ["long downward sweep", "post-drop release", "decaying fall", "smooth descent"]),
    "impact":    (["heavy downbeat hit", "cinematic boom", "deep slam", "braam transient"],
                  ["resonant tail", "short punchy decay", "booming sustain", "dramatic decay"]),
    "sweep":     (["white noise sweep", "filter whoosh", "airy motion", "noise transition"],
                  ["smooth sweep", "short whoosh", "rising-falling motion", "filtered tail"]),
    "texture":   (["vinyl crackle bed", "field-recording ambience", "granular texture", "foley layer", "glitch debris"],
                  ["evolving atmosphere", "static bed", "subtle movement", "lingering texture"]),
    "zap":       (["bright electric zap", "laser blip", "synthetic spark", "digital chirp"],
                  ["very short decay", "snappy tail", "quick blip", "tight decay"]),
    # pitched families
    "bass":      (["round low fundamental", "warm sub character", "gritty mid-bass", "tight punchy body"],
                  ["long natural decay", "smooth sustain", "slow decay", "controlled tail"]),
    "lead":      (["bright cutting tone", "detuned supersaw stack", "expressive single note", "punchy attack"],
                  ["sustained tone", "natural decay", "smooth sustain", "tight release"]),
    "pad":       (["lush evolving texture", "warm sustained tone", "wide stereo body", "soft airy character"],
                  ["slow attack sustain", "long evolving tail", "smooth sustained note", "gentle sustain"]),
    "key":       (["clear bell-like tone", "warm electric character", "percussive attack", "bright single note"],
                  ["natural decay", "medium sustain", "smooth release", "round decay"]),
    "pluck":     (["bright plucked attack", "woody single note", "crisp transient", "soft pluck"],
                  ["fast natural decay", "short ringing tail", "controlled decay", "quick release"]),
    "string":    (["bowed sustained tone", "warm ensemble character", "expressive single note", "rich body"],
                  ["smooth sustain", "long sustained note", "natural bow decay", "gentle release"]),
    "brass":     (["warm brassy tone", "bold single note", "round attack", "rich harmonic body"],
                  ["sustained tone", "natural decay", "smooth sustain", "controlled release"]),
    "wind":      (["breathy single note", "warm woodwind tone", "clear pitched note", "soft attack"],
                  ["sustained tone", "natural decay", "smooth sustain", "gentle release"]),
    "bell":      (["clear bell tone", "glassy strike", "metallic shimmer", "bright struck note"],
                  ["long ringing decay", "shimmering tail", "natural decay", "smooth fade"]),
    "mallet":    (["woody mallet strike", "warm struck tone", "bright percussive note", "soft mallet hit"],
                  ["natural decay", "short ringing tail", "quick decay", "round decay"]),
    "tperc":     (["tuned struck note", "resonant pitched hit", "clear pitched strike", "deep tuned tone"],
                  ["resonant decay", "ringing tail", "natural decay", "controlled sustain"]),
    "vocal":     (["clear sung note", "smooth vocal tone", "breathy single note", "expressive vowel"],
                  ["sustained note", "natural decay", "smooth sustain", "gentle release"]),
    "world":     (["bright plucked note", "resonant single note", "woody character", "expressive attack"],
                  ["natural decay", "short ringing tail", "controlled decay", "smooth release"]),
}

# Per-category: family, noun, exclusions, style preset. kind drum|pitched.
DRUM = "drum"
PITCH = "pitched"
CATS: dict[str, dict] = {
    # ---- drums / one-shots ----
    "kick":          dict(kind=DRUM, fam="kick", noun="kick drum", excl="no hi hats, no snare, no cymbals", w="balanced"),
    "snare-standard":dict(kind=DRUM, fam="snare", noun="snare drum", excl="no kick, no hi hats, no cymbals", w="balanced"),
    "snare-rim":     dict(kind=DRUM, fam="snare", noun="rimshot snare", excl="no kick, no hi hats, no cymbals, no long tail", w="balanced"),
    "hat-closed":    dict(kind=DRUM, fam="hat", noun="closed hi hat", excl="no kick, no snare, no cymbals, no sustained tone", w="balanced"),
    "hat-open":      dict(kind=DRUM, fam="hat", noun="open hi hat", excl="no kick, no snare, no crash", w="balanced"),
    "cymbal-ride":   dict(kind=DRUM, fam="cymbal", noun="ride cymbal", excl="no kick, no snare, no hi hats", w="balanced"),
    "cymbal-crash":  dict(kind=DRUM, fam="cymbal", noun="crash cymbal", excl="no kick, no snare, no hi hats", w="balanced"),
    "cymbal-splash": dict(kind=DRUM, fam="cymbal", noun="splash cymbal", excl="no kick, no snare, no hi hats", w="balanced"),
    "tamborine":     dict(kind=DRUM, fam="perc", noun="tambourine", excl="no kick, no snare, no cymbals", w="balanced"),
    "shaker":        dict(kind=DRUM, fam="perc", noun="shaker", excl="no kick, no snare, no cymbals, no sustained tone", w="balanced"),
    "tom-hi":        dict(kind=DRUM, fam="tom", noun="high tom", excl="no kick, no snare, no cymbals, no tuned drum", w="balanced"),
    "tom-mid":       dict(kind=DRUM, fam="tom", noun="mid tom", excl="no kick, no snare, no cymbals, no tuned drum", w="balanced"),
    "tom-low":       dict(kind=DRUM, fam="tom", noun="low tom", excl="no kick, no snare, no cymbals, no tuned drum", w="balanced"),
    "hit":           dict(kind=DRUM, fam="impact", noun="percussive stab hit", excl="no riser, no sweep, no melody", w="edm_heavy"),
    "clap":          dict(kind=DRUM, fam="clap", noun="clap", excl="no kick, no snare, no hi hats, no sustained tone", w="edm_heavy"),
    "808":           dict(kind=DRUM, fam="sub", noun="808 sub", excl="no hi hats, no snare, no kick click, no melody", w="edm_heavy"),
    "riser":         dict(kind=DRUM, fam="riser", noun="riser uplifter", excl="no drums, no drop, no melody", w="edm_heavy"),
    "downlifter":    dict(kind=DRUM, fam="downer", noun="downlifter", excl="no drums, no riser, no melody", w="edm_heavy"),
    "impact":        dict(kind=DRUM, fam="impact", noun="impact hit", excl="no riser, no sweep, no melody", w="edm_heavy"),
    "sub-drop":      dict(kind=DRUM, fam="sub", noun="sub drop", excl="no hi hats, no snare, no mid frequencies, no click", w="edm_heavy"),
    "sweep":         dict(kind=DRUM, fam="sweep", noun="noise sweep", excl="no drums, no melody, no tonal pitch", w="edm_heavy"),
    "texture":       dict(kind=DRUM, fam="texture", noun="texture atmosphere", excl="no drums, no melody, no clear pitch", w="edm_heavy"),
    "zap":           dict(kind=DRUM, fam="zap", noun="zap", excl="no drums, no melody, no sustained tone, no long decay", w="edm_heavy"),
    "foley-perc":    dict(kind=DRUM, fam="perc", noun="foley percussion hit", excl="no kick, no snare, no cymbals, no melody", w="urban"),
    # ---- pitched ----
    "synths":        dict(kind=PITCH, fam="lead", noun="synth", excl="no chord, no melody", w="edm_heavy"),
    "lead-supersaw": dict(kind=PITCH, fam="lead", noun="supersaw lead", excl="no chord, no melody", w="edm_heavy"),
    "lead-fm":       dict(kind=PITCH, fam="lead", noun="fm synth lead", excl="no chord, no melody", w="edm_heavy"),
    "lead-acid":     dict(kind=PITCH, fam="lead", noun="acid 303 lead", excl="no chord, no melody", w="edm_heavy"),
    "pluck-synth":   dict(kind=PITCH, fam="pluck", noun="synth pluck", excl="no chord, no melody", w="edm_heavy"),
    "plucks":        dict(kind=PITCH, fam="pluck", noun="acoustic pluck", excl="no chord, no melody", w="balanced"),
    "keys":          dict(kind=PITCH, fam="key", noun="electric keys", excl="no chord, no melody", w="urban"),
    "pianos":        dict(kind=PITCH, fam="key", noun="piano", excl="no chord, no melody", w="balanced"),
    "organs":        dict(kind=PITCH, fam="pad", noun="organ", excl="no chord, no melody", w="urban"),
    "basses":        dict(kind=PITCH, fam="bass", noun="bass", excl="no chord, no melody", w="urban"),
    "808-bass":      dict(kind=PITCH, fam="sub", noun="808 sub bass", excl="no chord, no melody", w="edm_heavy"),
    "reese-bass":    dict(kind=PITCH, fam="bass", noun="reese bass", excl="no chord, no melody", w="edm_heavy"),
    "pads":          dict(kind=PITCH, fam="pad", noun="pad", excl="no chord, no melody", w="balanced"),
    "strings":       dict(kind=PITCH, fam="string", noun="strings", excl="no chord, no melody", w="acoustic"),
    "brass":         dict(kind=PITCH, fam="brass", noun="brass", excl="no chord, no melody", w="acoustic"),
    "winds":         dict(kind=PITCH, fam="wind", noun="woodwind", excl="no chord, no melody", w="acoustic"),
    "accordion":     dict(kind=PITCH, fam="wind", noun="accordion", excl="no chord, no melody", w="acoustic"),
    "bells":         dict(kind=PITCH, fam="bell", noun="bell", excl="no chord, no melody", w="balanced"),
    "mallets":       dict(kind=PITCH, fam="mallet", noun="mallet instrument", excl="no chord, no melody", w="balanced"),
    "percussion":    dict(kind=PITCH, fam="tperc", noun="tuned percussion", excl="no chord, no melody, no untuned drum", w="balanced"),
    "timpani":       dict(kind=PITCH, fam="tperc", noun="timpani", excl="no chord, no melody, no untuned drum", w="acoustic"),
    "guitars":       dict(kind=PITCH, fam="pluck", noun="guitar", excl="no chord, no melody", w="balanced"),
    "banjos":        dict(kind=PITCH, fam="world", noun="banjo", excl="no chord, no melody", w="acoustic"),
    "mandolin":      dict(kind=PITCH, fam="world", noun="mandolin", excl="no chord, no melody", w="acoustic"),
    "harp":          dict(kind=PITCH, fam="world", noun="harp", excl="no chord, no melody", w="acoustic"),
    "sitar":         dict(kind=PITCH, fam="world", noun="sitar", excl="no chord, no melody", w="acoustic"),
    "vocals":        dict(kind=PITCH, fam="vocal", noun="vocal", excl="no chord, no melody, no instrumental backing", w="balanced"),
    "choir":         dict(kind=PITCH, fam="vocal", noun="choir", excl="no melody, no instrumental backing", w="acoustic"),
}


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _tokset(s: str) -> set:
    return set(_norm(s).replace(",", " ").split())


def _too_similar(cand: str, seen_tokens: list[set], thresh: float = 0.85) -> bool:
    ct = _tokset(cand)
    if not ct:
        return True
    for st in seen_tokens:
        inter = len(ct & st)
        union = len(ct | st)
        if union and inter / union >= thresh:
            return True
    return False


def _style(rng: random.Random, weights: dict) -> str:
    pool = rng.choices(list(weights), weights=list(weights.values()))[0]
    return rng.choice(STYLE_POOLS[pool])


def _pool_of(style: str) -> str:
    for pool, items in STYLE_POOLS.items():
        if style in items:
            return pool
    return "edm"


def gen_for_category(cat: str, spec: dict, target: int, existing: list[str]) -> tuple[list[str], dict]:
    """Return (new_lines, style_histogram). Preserves `existing` (counts toward
    target); only NEW lines are returned."""
    rng = random.Random(f"sas-v3:{cat}")
    fam_char, fam_decay = FAMILY[spec["fam"]]
    weights = WEIGHT_PRESETS[spec.get("w", "balanced")]
    noun, excl, kind = spec["noun"], spec["excl"], spec["kind"]

    seen = set(_norm(x) for x in existing)
    seen_tokens = [_tokset(x) for x in existing]
    new: list[str] = []
    hist = {"edm": 0, "urban": 0, "acoustic": 0}  # counts GENERATED lines only

    need = max(0, target - len(existing))
    attempts = 0
    while len(new) < need and attempts < need * 60:
        attempts += 1
        style = _style(rng, weights)
        tone = rng.choice(TONE)
        char = rng.choice(fam_char)
        decay = rng.choice(fam_decay)
        proc = rng.choice(PROCESSING)
        tmpl = rng.randint(0, 2)
        if tmpl == 0:
            line = f"{style} {tone} {noun} one shot, {char}, {decay}, dry, {excl}, no loop"
        elif tmpl == 1:
            line = f"{tone} {noun} one shot, {style} character, {char}, {proc}, {excl}, no loop"
        else:
            line = f"{style} {noun} one shot, {char}, {decay}, {proc}, {excl}, no loop"
        if kind == PITCH and "single note" not in line:
            line = line.replace("one shot,", "one shot, single note,", 1)
        n = _norm(line)
        if n in seen or _too_similar(line, seen_tokens):
            continue
        seen.add(n)
        seen_tokens.append(_tokset(line))
        new.append(line)
        hist[_pool_of(style)] += 1
    return new, hist


def read_existing(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def write_file(path: Path, cat: str, spec: dict, existing: list[str], new: list[str], hist: dict) -> None:
    total = len(existing) + len(new)
    edm_pct = round(100 * hist["edm"] / max(1, sum(hist.values())))
    lines = [
        "# " + "=" * 73,
        f"# {cat} — {'pitched single notes' if spec['kind'] == PITCH else 'unpitched one-shots'}. ~{total} prompts.",
        f"# Style mix (generated lines): ~{edm_pct}% EDM/electronic. EDM-forward, full spectrum.",
        f"# Generated by gen_prompts.py (deterministic). Curated lines preserved on top.",
        "# " + "=" * 73,
        "",
    ]
    if existing:
        lines.append("# --- curated ---")
        lines.extend(existing)
        lines.append("")
    if new:
        lines.append("# --- generated (combinatorial) ---")
        lines.extend(new)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=200, help="prompts per category")
    ap.add_argument("--only", nargs="*", default=None, help="subset of categories")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()

    cats = args.only or list(CATS)
    grand = 0
    for cat in cats:
        spec = CATS.get(cat)
        if spec is None:
            print(f"  ! unknown category {cat!r}, skipping")
            continue
        sub = "pitched/" if spec["kind"] == PITCH else ""
        path = REPO / "prompts" / sub / f"{cat}.txt"
        existing = read_existing(path)
        new, hist = gen_for_category(cat, spec, args.target, existing)
        total = len(existing) + len(new)
        grand += total
        print(f"  {cat:16} curated={len(existing):3d} +new={len(new):3d} = {total:3d}  "
              f"(edm {hist['edm']}/urban {hist['urban']}/acoustic {hist['acoustic']})")
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_file(path, cat, spec, existing, new, hist)
    print(f"\n{'(dry-run) ' if args.dry_run else ''}{len(cats)} categories, {grand} prompts total")


if __name__ == "__main__":
    main()
