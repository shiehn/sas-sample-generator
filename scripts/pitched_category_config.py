"""Per-category configuration for the pitched-instrument pipeline.

Sibling to category_config.py (drums). Each pitched category declares:

  - target_pitches_midi:  one or more MIDI notes the pipeline asks SA3 to
    aim for. Single-source categories have one (e.g. Plucks → C4 / 60).
    Multi-source categories (Phase 1.2) have two (e.g. Pianos → C2 + C4)
    so the per-zone resolver can pick the closer source at playback time.

  - duration_seconds:     length passed to SA3's seconds_total. Longer for
    sustaining categories (pads, organs) so we have a steady-state region
    to trim and loop in enrich. Shorter for plucks/percussion where the
    decay is the whole sample.

  - zone_span_semitones:  how far from each source pitch we render zones.
    +/- 12 (full octave) is the default; FX uses 0 (no shift at all).

  - zone_step_semitones:  pre-render granularity inside the span. 2 for
    formant-sensitive categories (vocals/pads/strings/pianos), 3 elsewhere.
    Tracktion does the in-between semitones via SRC at playback time.

  - pitch_tolerance_cents: max deviation from target pitch the gate
    accepts. 50 default; FX uses 9999 to skip the pitch gate entirely
    (sound design content, pitch is not the point).

  - min_sustain_seconds:  the sustain-plateau gate threshold. Short for
    pluck/percussion (0.2-0.6); long for pads (2.5+). Anything below
    rejects as "short stab" — catches the case where SA3 interprets a
    'sustained note' prompt as a percussive stab.

  - open_ended:           the manifest's open_ended flag, set on every
    zone. True for sustaining categories (Pads, Strings, Organs, Brass,
    Winds) so the Tracktion sampler plays for note-hold-duration; false
    for plucks/mallets/percussion which play through to end-of-sample.

  - pitch_detection_floor_hz: CREPE's reliable range is ~50 Hz up.
    Sub-bass categories (Basses target E1) need pyin cross-check below
    that. The 3-way agreement in gate_pitched takes over when target
    pitch is below ~82 Hz (E2).

  - skip_pitch_shift:     True only for FX. Zones never get pre-rendered;
    a single zone covers all keys at native pitch.

  - variants_per_prompt:  how many SA3 candidates per (prompt × target)
    we generate. The gate picks the best by composite score. Bumped to
    20 for Vocals because SA3's model card says vocals are weak — we
    need more attempts to find a usable one.

Phase 1.0 ships single-source for all categories except FX. Phase 1.2
flips Basses (E1+E2), Pianos (C2+C4), Strings (D3+A3), Winds (D3+A4)
to multi-source — uncomment the second pitch in target_pitches_midi.
"""

from dataclasses import dataclass


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
    variants_per_prompt: int = 5


# Negative prompts pattern: exclude "loop, melody, multiple notes, chord,
# reverb wash, vocals, drums" plus other categories that confuse SA3
# into emitting wrong-shape content. Vocals' negative does NOT exclude
# "vocals" (would suppress the target) and instead lists drums/melody.
_GENERIC_NEG = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "multiple notes, chord, melody, rhythmic pattern, reverb wash, "
    "long ambience, vocals, drums"
)

PITCHED_CATEGORIES: dict[str, PitchedCategoryConfig] = {
    "plucks":     PitchedCategoryConfig((60,),  _GENERIC_NEG, 3.0,  12, 3, 50, 0.4,  False, 80.0),
    "basses":     PitchedCategoryConfig((40,),  _GENERIC_NEG, 6.0,  12, 3, 50, 1.5,  False, 30.0),
    "bells":      PitchedCategoryConfig((72,),  _GENERIC_NEG, 4.0,  12, 3, 50, 0.8,  False, 200.0),
    "brass":      PitchedCategoryConfig((57,),  _GENERIC_NEG, 6.0,  12, 3, 50, 1.5,  True,  80.0),
    "fx":         PitchedCategoryConfig((69,),  _GENERIC_NEG, 4.0,  0,  1, 9999, 0.0, False, 80.0, skip_pitch_shift=True),
    "guitars":    PitchedCategoryConfig((52,),  _GENERIC_NEG, 4.0,  12, 3, 50, 0.6,  False, 80.0),
    "keys":       PitchedCategoryConfig((48,),  _GENERIC_NEG, 5.0,  12, 3, 50, 1.0,  False, 80.0),
    "mallets":    PitchedCategoryConfig((60,),  _GENERIC_NEG, 3.0,  12, 3, 50, 0.3,  False, 200.0),
    "organs":     PitchedCategoryConfig((48,),  _GENERIC_NEG, 8.0,  12, 3, 50, 2.0,  True,  80.0),
    "pads":       PitchedCategoryConfig((48,),  _GENERIC_NEG, 12.0, 12, 2, 50, 2.5,  True,  80.0),
    "pianos":     PitchedCategoryConfig((60,),  _GENERIC_NEG, 5.0,  12, 2, 50, 1.0,  False, 60.0),
    "percussion": PitchedCategoryConfig((60,),  _GENERIC_NEG, 2.0,  12, 3, 50, 0.2,  False, 80.0),
    "strings":    PitchedCategoryConfig((57,),  _GENERIC_NEG, 8.0,  12, 2, 50, 2.0,  True,  60.0),
    "synths":     PitchedCategoryConfig((48,),  _GENERIC_NEG, 5.0,  12, 3, 50, 1.5,  False, 60.0),
    "vocals":     PitchedCategoryConfig((57,),  _GENERIC_NEG, 5.0,  12, 2, 50, 1.5,  False, 100.0, variants_per_prompt=20),
    "winds":      PitchedCategoryConfig((69,),  _GENERIC_NEG, 5.0,  12, 2, 50, 1.5,  True,  100.0),
}


# Phase 1.2 multi-source overrides (uncomment to flip on):
#
# PITCHED_CATEGORIES["basses"]  = replace(PITCHED_CATEGORIES["basses"],  target_pitches_midi=(28, 40))   # E1 + E2
# PITCHED_CATEGORIES["pianos"]  = replace(PITCHED_CATEGORIES["pianos"],  target_pitches_midi=(36, 60))   # C2 + C4
# PITCHED_CATEGORIES["strings"] = replace(PITCHED_CATEGORIES["strings"], target_pitches_midi=(50, 57))   # D3 + A3
# PITCHED_CATEGORIES["winds"]   = replace(PITCHED_CATEGORIES["winds"],   target_pitches_midi=(50, 69))   # D3 + A4
