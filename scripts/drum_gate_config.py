"""Per-category quality profiles for the drum / one-shot gate (gate_drums.py).

Drums are unpitched, so the gate can't lean on pitch like the pitched pipeline.
Instead it checks, per category PROFILE:

  - kind:          percussive | cymbal | sub | fx  (selects which gates apply)
  - centroid_hz:   the expected spectral-centroid band. Used as a SOFT score
                   (prefer on-band variants) and a LENIENT hard reject only when
                   a sample is grossly off-band (e.g. an all-treble "kick").
  - max_onset_ms:  attack must land within this from the start (percussive /
                   cymbal / sub). 0 disables (fx build/sweep — no fast transient).
  - expect_decay:  the tail must fall well below the peak (reject sustained
                   drones / pads where a one-shot was asked for). False for fx.
  - single_hit:    reject multiple spaced transients (a loop / roll leaked in).
                   False for rattly content (shaker/tambourine) and fx.
  - min_duration_s:reject if shorter (truncated / clipped-off sample).
  - variants_per_prompt: how many SA3 candidates the gate picks the best of.
                   Higher for the highest-value cats (kick/snare/clap/808).

A category with no entry falls back to DEFAULT_DRUM_PROFILE (lenient percussive).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DrumProfile:
    kind: str
    centroid_hz: tuple[float, float]
    max_onset_ms: float
    expect_decay: bool
    single_hit: bool
    min_duration_s: float
    variants_per_prompt: int


def _p(kind, lo, hi, onset, decay, single, min_dur, var):
    return DrumProfile(kind, (lo, hi), onset, decay, single, min_dur, var)


DRUM_PROFILES: dict[str, DrumProfile] = {
    # category        kind          centroid Hz   onset  decay  single  min_s  var
    "kick":         _p("sub",         20, 600,     90,   True,  True,   0.15,  6),
    "snare-standard": _p("percussive", 700, 6000,  70,   True,  True,   0.12,  6),
    "snare-rim":    _p("percussive", 1000, 8000,   50,   True,  True,   0.05,  6),
    "hat-closed":   _p("percussive", 4000, 16000,  40,   True,  True,   0.03,  5),
    "hat-open":     _p("percussive", 3500, 15000,  60,   True,  False,  0.10,  5),
    "cymbal-ride":  _p("cymbal",     3000, 14000, 120,   True,  False,  0.40,  5),
    "cymbal-crash": _p("cymbal",     3000, 14000, 120,   True,  False,  0.50,  5),
    "cymbal-splash":_p("cymbal",     3500, 15000, 100,   True,  False,  0.25,  5),
    "tamborine":    _p("percussive", 4000, 13000,  90,   True,  False,  0.10,  5),
    "shaker":       _p("percussive", 3000, 13000, 120,   True,  False,  0.08,  5),
    "tom-hi":       _p("percussive",  200, 1500,   80,   True,  True,   0.12,  5),
    "tom-mid":      _p("percussive",  150, 1100,   80,   True,  True,   0.15,  5),
    "tom-low":      _p("percussive",   80,  800,   90,   True,  True,   0.18,  5),
    "hit":          _p("percussive",   80, 6000,  120,   True,  True,   0.15,  5),
    "clap":         _p("percussive", 1000, 7000,   90,   True,  True,   0.08,  6),
    "808":          _p("sub",          20,  500,  110,   True,  True,   0.30,  6),
    "sub-drop":     _p("sub",          20,  400,    0,   True,  False,  0.40,  4),
    "riser":        _p("fx",          200, 16000,   0,   False, False,  1.50,  4),
    "downlifter":   _p("fx",          200, 16000,   0,   False, False,  1.00,  4),
    "impact":       _p("percussive",   50, 4000,  120,   True,  True,   0.20,  5),
    "sweep":        _p("fx",          500, 16000,   0,   False, False,  0.80,  4),
    "texture":      _p("fx",          200, 14000,   0,   False, False,  1.00,  4),
    "zap":          _p("percussive", 1000, 14000,  40,   True,  True,   0.03,  5),
    "foley-perc":   _p("percussive",  400, 9000,   90,   True,  True,   0.04,  5),
}

DEFAULT_DRUM_PROFILE = _p("percussive", 80, 12000, 120, True, False, 0.05, 5)


def get_profile(category: str) -> DrumProfile:
    return DRUM_PROFILES.get(category, DEFAULT_DRUM_PROFILE)
