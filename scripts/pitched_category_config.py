"""Per-category configuration for the pitched-instrument pipeline (v3).

Each pitched category declares:

  - target_pitches_midi:  one or more MIDI notes SA3 is asked to aim for. The
    list_to_jsonl fanout emits one job per (prompt × target_pitch); enrich then
    MERGES the surviving pitches of one prompt into a single multi-source
    instrument (each zone rendered from its nearest real source -> small,
    artifact-free pitch shifts across the keyboard).

  - duration_seconds:     SA3 seconds_total. Longer for sustaining categories so
    enrich has a steady-state region to trim/loop.

  - zone_span_semitones:  how far from each source pitch zones are rendered.
    12 for single-source categories; 7 for multi-source (the real sources cover
    the rest, and the smaller span drops the worst extreme-shift zones).

  - zone_step_semitones:  pre-render granularity. 2 for formant-sensitive
    categories (vocals/choir/pads/strings/pianos/winds), 3 elsewhere. Tracktion
    SRC-interpolates the in-between semitones at playback.

  - pitch_tolerance_cents: 9999 (off) for everything — SA3 can't reliably hit a
    prompted pitch, so enrich measures the actual pitch and snaps to the nearest
    semitone. The gate only confirms "this is a clearly-pitched note".

  - min_sustain_seconds:  sustain-plateau gate threshold (rejects percussive
    stabs where a sustained note was asked for).

  - open_ended:           manifest open_ended flag. True for sustaining
    categories (pads/strings/organs/brass/winds/choir/accordion).

  - pitch_detection_floor_hz: pyin/autocorr lower bound hint for sub-bass.

  - variants_per_prompt:  SA3 candidates per (prompt × pitch); the gate keeps
    the best. Higher for hard/weak categories (sub-bass, sustained, vocals).

MULTI-SOURCE TOGGLE
-------------------
SAS_MULTI_SOURCE=1 (default) flips the wide-range categories to multiple real
source pitches + a tighter zone_span. SAS_MULTI_SOURCE=0 reproduces the v1
single-source behaviour (one root, span 12) for an A/B comparison.
"""

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PitchedCategoryConfig:
    target_pitches_midi: tuple[int, ...]
    negative_prompt: str
    duration_seconds: float
    zone_span_semitones: int
    zone_step_semitones: int
    pitch_tolerance_cents: int
    min_sustain_seconds: float
    open_ended: bool
    pitch_detection_floor_hz: float
    skip_pitch_shift: bool = False
    # How aggressively enrich shifts toward the prompted target: within N
    # semitones -> lock to target (preserves prompt semantics); else snap to the
    # nearest integer semitone (<50c, no artifact) and use THAT as the root.
    max_correction_semitones: int = 3
    variants_per_prompt: int = 6


# Generic negative: suppress phrasing/percussion/loops so SA3 emits a single
# sustained note. Vocal categories use _VOCAL_NEG so we don't suppress the target.
_GENERIC_NEG = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "multiple notes, chord, melody, rhythmic pattern, reverb wash, "
    "long ambience, vocals, drums"
)
_VOCAL_NEG = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "multiple notes, chord, melody, rhythmic pattern, reverb wash, "
    "long ambience, drums, instrumental backing"
)


def _cat(roots, dur, *, span=12, step=3, min_sus=0.2, open_ended=False,
         floor=80.0, var=6, neg=_GENERIC_NEG, tol=9999, max_corr=3,
         skip_shift=False) -> PitchedCategoryConfig:
    """Compact, keyword-driven constructor so the 28-entry table stays readable
    and position-error-proof."""
    return PitchedCategoryConfig(
        target_pitches_midi=tuple(roots),
        negative_prompt=neg,
        duration_seconds=dur,
        zone_span_semitones=span,
        zone_step_semitones=step,
        pitch_tolerance_cents=tol,
        min_sustain_seconds=min_sus,
        open_ended=open_ended,
        pitch_detection_floor_hz=floor,
        skip_pitch_shift=skip_shift,
        max_correction_semitones=max_corr,
        variants_per_prompt=var,
    )


# Base table: SINGLE-source roots, span 12 (== SAS_MULTI_SOURCE=0 behaviour).
# Multi-source categories get expanded below when the toggle is on.
PITCHED_CATEGORIES: dict[str, PitchedCategoryConfig] = {
    # --- keys / synths / leads (EDM-forward) ---
    "synths":        _cat((48,), 5.0, min_sus=0.5, floor=60, var=6),
    "lead-supersaw": _cat((60,), 5.0, min_sus=0.4, floor=80, var=6),
    "lead-fm":       _cat((60,), 5.0, min_sus=0.3, floor=80, var=6),
    "lead-acid":     _cat((48,), 4.0, min_sus=0.3, floor=60, var=6),
    "pluck-synth":   _cat((60,), 3.0, min_sus=0.1, floor=80, var=6),
    "plucks":        _cat((60,), 3.0, min_sus=0.1, floor=80, var=6),
    "keys":          _cat((48,), 5.0, min_sus=0.4, floor=80, var=6),
    "pianos":        _cat((60,), 5.0, step=2, min_sus=0.4, floor=60, var=6),
    "organs":        _cat((48,), 8.0, min_sus=0.75, open_ended=True, floor=80, var=8),
    # --- bass family ---
    "basses":        _cat((40,), 6.0, min_sus=0.5, floor=30, var=8),
    "808-bass":      _cat((36,), 6.0, min_sus=0.5, floor=25, var=8),
    "reese-bass":    _cat((40,), 6.0, min_sus=0.4, floor=35, var=8),
    # --- pads / strings / brass / winds (sustaining) ---
    "pads":          _cat((48,), 12.0, step=2, min_sus=1.0, open_ended=True, floor=80, var=8),
    "strings":       _cat((57,), 8.0, step=2, min_sus=0.75, open_ended=True, floor=60, var=8),
    "brass":         _cat((57,), 6.0, min_sus=0.5, open_ended=True, floor=80, var=8),
    "winds":         _cat((69,), 5.0, step=2, min_sus=0.5, open_ended=True, floor=100, var=8),
    "accordion":     _cat((54,), 6.0, min_sus=0.6, open_ended=True, floor=80, var=8),
    # --- bells / mallets / tonal percussion ---
    "bells":         _cat((72,), 4.0, min_sus=0.3, floor=200, var=6),
    "mallets":       _cat((60,), 3.0, min_sus=0.1, floor=200, var=6),
    "percussion":    _cat((60,), 2.0, min_sus=0.08, floor=80, var=6),
    "timpani":       _cat((48,), 4.0, min_sus=0.3, floor=35, var=8),
    # --- plucked / world / acoustic ---
    "guitars":       _cat((52,), 4.0, min_sus=0.2, floor=80, var=6),
    "banjos":        _cat((60,), 3.0, min_sus=0.1, floor=90, var=6),
    "mandolin":      _cat((69,), 3.0, min_sus=0.12, floor=130, var=6),
    "harp":          _cat((60,), 4.0, min_sus=0.15, floor=60, var=6),
    "sitar":         _cat((60,), 4.0, min_sus=0.2, floor=90, var=6),
    # --- vocal ---
    "vocals":        _cat((57,), 5.0, step=2, min_sus=0.5, floor=100, var=20, neg=_VOCAL_NEG),
    "choir":         _cat((57,), 10.0, step=2, min_sus=1.0, open_ended=True, floor=90, var=18, neg=_VOCAL_NEG),
}


# Multi-source expansion (wide-range instruments). Each gets 2-4 REAL source
# pitches spanning its natural register + a tighter zone_span (7) so no zone is
# pitch-shifted more than ~half the inter-source gap. zone_step is preserved.
# Toggle off (SAS_MULTI_SOURCE=0) to A/B against single-source.
_MULTI_SOURCE_ROOTS: dict[str, tuple[int, ...]] = {
    "basses":        (28, 40, 52),   # E1, E2, E3
    "808-bass":      (36, 48),       # C2, C3
    "reese-bass":    (40, 52),       # E2, E3
    "pianos":        (36, 48, 60, 72),  # C2..C5 (timbre varies most across registers)
    "strings":       (45, 57, 69),   # A2, A3, A4
    "brass":         (45, 57, 69),   # A2, A3, A4 (tuba..trumpet)
    "winds":         (50, 62, 74),   # D3, D4, D5
    "guitars":       (40, 52, 64),   # E2, E3, E4
    "harp":          (48, 60, 72),   # C3, C4, C5
    "timpani":       (41, 53),       # F2, F3 (tuned drum — low roots matter)
    "lead-supersaw": (60, 72),       # C4, C5
    "banjos":        (60, 67),       # C4, G4
}

MULTI_SOURCE = os.environ.get("SAS_MULTI_SOURCE", "1") == "1"
if MULTI_SOURCE:
    for _name, _roots in _MULTI_SOURCE_ROOTS.items():
        if _name in PITCHED_CATEGORIES:
            PITCHED_CATEGORIES[_name] = replace(
                PITCHED_CATEGORIES[_name],
                target_pitches_midi=_roots,
                zone_span_semitones=7,
            )
