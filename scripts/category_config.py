"""Per-category configuration for the drum / unpitched one-shot generator (v3).

Each category gets its own negative prompt — what the model should NOT generate.
The trick: the negative prompt must exclude OTHER drum types, not the target
sound. A hat category that lists "hi hats" in its negative would avoid the very
thing we asked for.

v3 adds an EDM/electronic one-shot family carved out of the old catch-all `hit`
and the retired pitched `fx`: clap, 808, riser, downlifter, impact, sub-drop,
sweep, texture, zap, foley-perc. `hit` is narrowed to generic stabs. Toms now
exclude *tuned* drums so they don't bleed into the pitched `timpani`/`percussion`
categories. The old drum `plucks` stub is removed (plucks are pitched).

Durations are tuned per category: short for clicks/claps/zaps, long for cymbals
and transition FX with tails.
"""

CATEGORY_NEGATIVES: dict[str, str] = {
    # ---- core kit ----
    "kick": (
        "low quality, distorted, noisy, clipped, music loop, drum loop, "
        "hi hats, snare, cymbals, toms, vocals, melody, rhythmic pattern, "
        "reverb wash, long ambience"
    ),
    "snare-standard": (
        "low quality, music loop, drum loop, kick, hi hats, cymbals, toms, "
        "vocals, melody, reverb wash, long ambience"
    ),
    "snare-rim": (
        "low quality, music loop, drum loop, kick, hi hats, cymbals, toms, "
        "vocals, melody, long decay, reverb wash, sustained tone"
    ),
    "hat-closed": (
        "low quality, music loop, drum loop, kick, snare, toms, ride cymbal, "
        "crash cymbal, vocals, melody, long reverb, sustained tone"
    ),
    "hat-open": (
        "low quality, music loop, drum loop, kick, snare, toms, ride cymbal, "
        "crash cymbal, vocals, melody, sustained tone"
    ),
    "cymbal-ride": (
        "low quality, music loop, drum loop, kick, snare, hi hats, toms, "
        "vocals, melody, short decay, rhythmic pattern"
    ),
    "cymbal-crash": (
        "low quality, music loop, drum loop, kick, snare, hi hats, toms, "
        "vocals, melody, short decay"
    ),
    "cymbal-splash": (
        "low quality, music loop, drum loop, kick, snare, hi hats, toms, "
        "vocals, melody, long sustain"
    ),
    "tamborine": (
        "low quality, music loop, drum loop, kick, snare, cymbals, toms, "
        "vocals, melody, long ambience, rhythmic pattern"
    ),
    "shaker": (
        "low quality, music loop, drum loop, kick, snare, cymbals, toms, "
        "vocals, melody, rhythmic pattern, sustained tone"
    ),
    # Toms exclude TUNED drums so they don't overlap pitched timpani/percussion.
    "tom-hi": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash, tuned drum, defined pitch, timpani"
    ),
    "tom-mid": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash, tuned drum, defined pitch, timpani"
    ),
    "tom-low": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash, tuned drum, defined pitch, timpani"
    ),
    # ---- generic stab (narrowed) ----
    "hit": (
        "low quality, music loop, drum loop, vocals, melody, long ambience, "
        "rhythmic pattern, sustained tone, riser, sweep, build-up"
    ),
    # ---- EDM / electronic one-shot family ----
    "clap": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "toms, melody, vocals, sustained tone, reverb wash"
    ),
    "808": (
        # The 808 IS the tuned sub — don't exclude low pitch; exclude the click
        # transient (that's the kick) and any phrasing.
        "low quality, music loop, drum loop, hi hats, snare, cymbals, "
        "kick transient click, melody, chord, multiple notes, vocals"
    ),
    "riser": (
        "low quality, music loop, drum loop, drums, kick, snare, "
        "downward sweep, impact, drop, melody, rhythmic pattern, vocals"
    ),
    "downlifter": (
        "low quality, music loop, drum loop, drums, upward sweep, riser, "
        "build-up, melody, rhythmic pattern, vocals"
    ),
    "impact": (
        "low quality, music loop, drum loop, riser, sweep, build-up, "
        "melody, rhythmic pattern, vocals"
    ),
    "sub-drop": (
        "low quality, music loop, drum loop, hi hats, snare, cymbals, "
        "melody, mid frequencies, high frequencies, transient click, vocals"
    ),
    "sweep": (
        "low quality, music loop, drum loop, drums, melody, "
        "rhythmic pattern, tonal pitch, chord, vocals"
    ),
    "texture": (
        "low quality, music loop, drum loop, drums, melody, rhythmic pattern, "
        "clear pitch, musical note, vocals"
    ),
    "zap": (
        "low quality, music loop, drum loop, drums, melody, "
        "sustained tone, reverb wash, long decay, vocals"
    ),
    "foley-perc": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "melody, vocals, sustained tone, reverb wash"
    ),
}

CATEGORY_DURATIONS: dict[str, float] = {
    # core kit
    "kick": 1.5,
    "snare-standard": 1.0,
    "snare-rim": 0.75,
    "hat-closed": 0.5,
    "hat-open": 1.5,
    "cymbal-ride": 2.5,
    "cymbal-crash": 3.0,
    "cymbal-splash": 1.5,
    "tamborine": 1.0,
    "shaker": 0.75,
    "tom-hi": 1.0,
    "tom-mid": 1.25,
    "tom-low": 1.5,
    "hit": 1.5,
    # EDM / electronic one-shots
    "clap": 0.75,
    "808": 2.0,
    "riser": 4.0,
    "downlifter": 3.0,
    "impact": 2.0,
    "sub-drop": 2.0,
    "sweep": 2.5,
    "texture": 3.0,
    "zap": 0.75,
    "foley-perc": 0.75,
}

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "vocals, melody, long ambience, reverb wash"
)
