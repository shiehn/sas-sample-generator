"""Per-category configuration for the sample generator.

Each category gets its own negative prompt — what the model should NOT
generate. The trick: the negative prompt must exclude OTHER drum types,
not the target sound. A hat category that lists "hi hats" in its negative
prompt would actively avoid generating what we asked for.

Durations are tuned to the category: short for clicks and rim shots,
longer for cymbals with tails.
"""

CATEGORY_NEGATIVES: dict[str, str] = {
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
    "tom-hi": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash"
    ),
    "tom-mid": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash"
    ),
    "tom-low": (
        "low quality, music loop, drum loop, kick, snare, hi hats, cymbals, "
        "vocals, melody, reverb wash"
    ),
    "hit": (
        "low quality, music loop, drum loop, vocals, melody, long ambience, "
        "rhythmic pattern, sustained tone"
    ),
}

CATEGORY_DURATIONS: dict[str, float] = {
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
}

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "vocals, melody, long ambience, reverb wash"
)
