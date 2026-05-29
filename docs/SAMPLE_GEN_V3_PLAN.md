# Sample-Generation Run v3 — Findings & Implementation Plan

> Findings + recommendations + execution plan for the v3 sample-generation campaign. Authored
> 2026-05-28. Locked decisions are in **Scope decisions** below; implementation status is tracked
> in the team task list, not here.

## Context

We're doing another instrument/drum sample-generation run with a bigger GPU budget (up to $200).
Goals: a more powerful sample **collection** (and sampler where it pays off), **accurate pitches**,
**more categories** with an EDM/electronic bias but full-spectrum coverage, and **~150 samples per
category**. The pipeline synthesizes samples with **Stable Audio 3** on a cloud GPU (RunPod), gates
them for quality, enriches (pitch-correct + pre-render multi-zone instruments), and ships versioned
packs the Electron app (`sas-app`) downloads. Two consumers must stay compatible: the **drum plugin**
(flat one-shot folders, folder name = role, Tracktion single-sound sampler) and the **instrument
plugin** (manifest-based, multi-zone Tracktion sampler).

### Findings summary (answers to the original asks)
- **(a/b) What exists / how samples are used:** Two pipelines. Drums → `postprocess_oneshots.py` →
  flat `processed/<cat>/*.wav` (no gate). Pitched → `gate_pitched.py` (5-stage, spectral-fundamental
  pitch detection, ~70% pitch-class accuracy) → `enrich_pitched.py` (pitch-correct via `rubberband`,
  −20 LUFS, pre-render `zones/*.flac` ±12 semitones, write `manifest.json`). Drum plugin reads folder
  names; instrument plugin consumes **`zones`** from the manifest (the `sources[]` array is metadata).
- **(c/d) Stable Audio 3:** We already run `stabilityai/stable-audio-3-medium` (1.4B) — **the most
  powerful *open-weight* SA3 model**. SA3-Large (2.7B) is **API/enterprise-only, not self-hostable**,
  so "use the most powerful version" = use Medium *better* (batching, oversampling, multi-source),
  not swap models. Licensing: Stability AI Community License (commercial OK under $1M rev).
- **(j) Budget:** Compute is **not** the constraint — even a maximal run is far under $200. The real
  bottleneck is the **gate stage** (per-variant pitch detection). A bigger GPU mainly buys wall-clock
  via **batched generation**.
- **(k) Sampler:** The biggest quality lever — **multi-source real-pitch sampling** — needs **no
  sampler change** (plugin consumes pre-rendered `zones`; `sources[]` is already an array; "Phase 1.2"
  multi-source is wired but commented out). **Velocity layers / round-robin are out of scope**
  (Tracktion's built-in `SamplerPlugin` has no velocity dimension and "double-triggers every matching
  sound"; it's a git submodule, so extending = forking — rejected).
- **(k) Zone format:** Tracktion CPU-decodes FLAC on the message thread (stalls on load); WAV is mmap'd.
  Ship zones as **16-bit WAV** instead of 24-bit FLAC → removes the decode penalty for ~15% more
  download, no engine change (see Part 4 *Zone format*).

### Scope decisions (locked)
1. **Samples + multi-source only.** No engine/SDK/plugin changes. Manifest stays `schema_version: 1`.
2. **Full category restructure** → ~22 drum/one-shot + ~28 pitched ≈ **50 categories**.
3. **Single full run** (author all prompts, generate everything in one campaign, then review).
4. **One big instrument pack** (no per-category/tiered split, no app-side marker change). **150
   samples/category.** Contain size with zone-span reduction + **16-bit WAV zones**.

### Intended outcome
A noticeably more accurate (pitch-class accuracy target **≥80%**, up from 70%), broader, EDM-leaning
sample collection at ~150/category across ~50 categories, generated efficiently on a single big-GPU
RunPod campaign, shipped as bumped drum + instrument packs the app already knows how to consume.

---

## Part 1 — Pitch accuracy (top priority)

### 1a. Multi-source real-pitch sampling — the headline win (generator-only)
**Problem:** today each instrument is one recorded root pitch with `zones` pitch-shifted ±12 semitones
via rubberband; extreme shifts smear transients/formants and drift pitch.
**Fix:** record **2–3 real source pitches** spanning each category's natural range; enrich assigns each
zone to its **nearest real source** so per-zone shift stays small (≤ ~4 semitones).

- `scripts/pitched_category_config.py`: uncomment + **extend** `target_pitches_midi` (lines 124–129)
  per the table in Part 3. Gate behind `MULTI_SOURCE = os.environ.get("SAS_MULTI_SOURCE","1")=="1"`
  for a one-env-var A/B. **Cut `zone_span_semitones` 12→7** for multi-source categories (fewer rendered
  zones + drops the worst extreme-shift zones → also the primary pack-size lever, Part 4).
- `scripts/enrich_pitched.py` — the real work. Today single-source (`enrich_sample`, 351–541; effective
  root 391–413; zone build 448–483; writes one source 515–531). Add:
  1. **Group gated winners by `(category, normalized_prompt)`** across sibling pitches in `run_enrich`
     (284–348). The N target-pitch jobs for one prompt have distinct ids (pitch is hashed in), so
     recover the prompt from the gate.json/raw-meta and merge them into ONE instrument.
  2. New **`enrich_multi_source(group)`**: per source, run existing smart pitch-correction → write
     `sources/<root>.wav`; compute candidate zone roots (±`zone_span` step `zone_step`) and **assign
     each to the nearest source**; dedup; then build **disjoint, ordered `min/max_midi`** via the
     **existing midpoint-split (461–469)** over the union of roots. Render each zone as **16-bit WAV**
     from its assigned source (mmap-able — replaces FLAC; see Part 4 *Zone format*). Emit
     `sources: [one per source]` + merged `zones`. `schema_version: 1`, `open_ended` unchanged.
  3. **Pre-write assertion** that `zones` are strictly increasing and contiguous/non-overlapping
     (the engine double-triggers overlaps — load-bearing invariant). Fail → skip instrument.
- **Why safe (no plugin change):** `instrument-resolver.ts` maps `zones[].root_midi/min/max` straight
  to sampler zones and only checks disjoint+ordered+`schema_version===1`; extra `sources[]` are ignored.
- Guard `scripts/repair_instrument_pitch.py` (reads `sources[0]` only): if `len(sources)>1`, skip+warn.

### 1b. Oversample + best-variant selection
Gate already picks best-of-N by composite score (`gate_pitched.py:605,700`). Raise per-category
`variants_per_prompt` (config) so the tail improves:
- workhorses (pianos/keys/synths/guitars/bells/mallets/plucks/percussion/leads): **6**
- hard (basses/808/reese/organs/pads/strings/brass/winds, sub-bass + sustained): **8–10**
- vocals/choir: **16–20** (SA3 vocals are weak)
- one-shot FX categories: low (pitch doesn't matter) — handled in drum pipeline.
Make counts flow through generation: add a `"variants"` field per JSONL row in
`scripts/list_to_jsonl_pitched.py` (106–113) and drop the global `--num-waveforms-per-prompt` override
in `scripts/run_pitched.sh` (41,100) so per-category config wins.

### 1c. Ensemble octave cross-check (sub-bass robustness)
`_pyin_pitch` (343–347) and `_autocorr_pitch` (350–374) already exist but are unused. The spectral
detector nails pitch *class*; add an **octave-only** cross-check for sub-bass targets (≤ ~E2): if
pyin+autocorr agree on an octave that differs from spectral by ±12/±24, re-seat spectral to consensus
(preserve sub-semitone fraction). Wire into `gate_pitch` (432–436); record all three estimates in
`verdict["metrics"]`. Keep torchcrepe for voicing/confidence only. Restrict to sub-bass so gate
wall-clock barely moves; try/except → spectral-only fallback. Add octave-rescue unit tests to
`tests/test_pitch_detection.py` (reuse `_synth`, 28–43; pure-CPU).

### 1d. init_audio pitch anchoring (experiment — adopt only if it wins)
`generate_diffusion_cond_inpaint` already accepts `init_audio=(sr,tensor)` + `init_noise_level`.
Add `--init-audio-anchor` + `--init-noise-level` to `batch_generate.py`; synth a harmonic tone at the
row's `target_pitch_midi` (reuse `_synth` from the test) and pass it. **A/B before committing:** 2 hard
categories (basses, synths), sweep `init_noise_level ∈ {0.5,0.65,0.8}`, compare via the 1e report.
Adopt per-category only if median |cents| **and** pitch-class accuracy improve **without** raising the
all-variants-fail rate (a strong anchor collapses timbre variety). **Default off.** Note: init_audio
broadcasts across a batch, so anchored batches must be single-target-pitch (Part 2).

### 1e. Pitch-accuracy verification report (makes "better" measurable)
New `scripts/pitch_report.py` (read-only; writes under `outputs/_reports/`, already `_`-excluded from
packs). Walk `outputs/gated/<cat>/*.gate.json` (+ manifests post-enrich); emit per-category: signed
cents histogram, **pitch-class accuracy**, **octave accuracy**, median/p90 |cents|, all-variants-fail
rate, per-rejection-reason tally. Write `pitch_summary.{json,md}` so two runs diff directly. Add a
`report` stage to `run_pitched.sh` (after gate). This is the artifact the 1d A/B and the acceptance
gate (Part 6) are judged on.

---

## Part 2 — Batched generation (exploit the bigger GPU)
Today `generate_for_jsonl` (`batch_generate.py:45–116`) runs `batch_size=1` and clears CUDA cache every
sample. Rewrite to batch:
- Add `--batch-size` (default 16; 32–64 on 80GB). Build a flat work-list of `(id, variant)` units;
  apply `--skip-existing` by dropping units whose `wav_path` exists (preserves content-addressed skip).
- **Bucket by `duration_seconds`** (homogeneous `seconds_total` → no wasted denoising); if
  `--init-audio-anchor`, also bucket by target pitch. One `generate_diffusion_cond_inpaint` call per
  batch with `conditioning=[{prompt,seconds_total}...]`, full-length `negative_conditioning`,
  `batch_size=len(batch)`, one seed/batch; slice `output[i]`. Move `empty_cache()` to once/batch.
- **Cheap pre-gate on the batch tensor (GPU) before writing WAVs:** drop NaN/Inf/silent/clipped to cut
  disk + downstream gate load. (Most impactful gate-bottleneck mitigation.)
- Record `batch_size`/`batch_index` in metadata. Document: batch_size is part of run identity (changes
  noise for not-yet-generated variants); skip-existing keeps resumes stable.

**Throughput:** generation drops from many hours to ~1–3h. **Gate becomes the bottleneck** (torchcrepe +
basic-pitch + librosa per variant, unbatched). Mitigate via the pre-gate above, sub-bass-only ensemble,
batched crepe where possible, and per-category parallelism (mirror enrich's `ProcessPoolExecutor`).

---

## Part 3 — Category taxonomy (full restructure, ~50)

**Binding rule:** category ID = folder name everywhere; only validation is `^[a-z0-9-]+$`
(`list_to_jsonl*.py`). Adding a category = config entry + `prompts/.../<cat>.txt` + enable line in
`categories.txt`/`pitched_categories.txt`. Renames/splits ⇒ regenerate + pack version bump.

### A. Drum / unpitched one-shot (`category_config.py`, `categories.txt`, `prompts/<cat>.txt`)
KEEP: kick, snare-standard, snare-rim, hat-closed, hat-open, cymbal-ride, cymbal-crash, cymbal-splash,
tamborine, shaker, tom-hi, tom-mid, tom-low. NARROW `hit` to generic stabs.
**NEW (EDM-forward, carved from `hit`/retired pitched `fx`):** `clap`, `808` (flat tuned one-shot),
`riser`, `downlifter`, `impact`, `sub-drop`, `sweep`, `texture` (vinyl/foley/glitch), `zap`,
`foley-perc` (optional). **Remove** the spurious drum `plucks` stub (belongs to pitched).
Durations/negatives per the design table (claps 0.75s; 808 2.0s exclude "kick click"; riser 4s exclude
"downward sweep/drop"; texture exclude "clear pitch/musical note"; toms add "no tuned drum/timpani").
**~22 roles.**

### B. Pitched instruments (`pitched_category_config.py`, `pitched_categories.txt`, `prompts/pitched/`)
KEEP: plucks, basses (narrow), bells, brass, guitars, keys, mallets, organs, pads, pianos, percussion
(redefine = *tonal/tuned* only), strings, synths (narrow), vocals, winds. **Retire pitched `fx`** →
one-shots.
**NEW / split:** `808-bass`, `reese-bass`, `lead-supersaw`, `lead-fm`, `lead-acid`, `pluck-synth`,
`choir`, `banjos`, `mandolin`, `timpani` (tuned, low roots), `harp`, `accordion`, `sitar`. **~28
categories.**

**Render config (target roots / duration / open_ended / special) — key rows:**

| Category | roots (MIDI) | dur s | open_ended | special |
|---|---|---|---|---|
| basses | E1+E2 (28,40) ms | 6 | no | floor 30Hz |
| 808-bass | C2+C3 (36,48) | 6 | no | floor 25Hz |
| reese-bass | E2+E3 (40,52) | 6 | no | floor 35Hz |
| pianos | C2+C4 (36,60) ms | 5 | no | step 2 |
| strings | D3+A3 (50,57) ms | 8 | yes | step 2 |
| winds | D3+A4 (50,69) ms | 5 | yes | — |
| harp | C3+C5 (48,72) ms | 4 | no | wide |
| brass | A3+A4 (57,69) | 6 | yes | — |
| lead-supersaw | C4+C5 (60,72) | 5 | no | — |
| banjos | C4+G4 (60,67) | 3 | no | bright decay |
| timpani | F2+C3 (41,48) | 4 | no | tuned drum, floor 35Hz |
| choir | A3 (57) | 10 | yes | variants 20, step 2 |
| (plucks/bells/mallets/keys/organs/pads/synths/vocals/percussion/guitars/lead-fm/lead-acid/pluck-synth/sitar/accordion/mandolin) | per design table | — | — | mostly single/2-root |

`ms` = multi-source. `zone_span` 12→7 and `zone_step` 2/3 unchanged otherwise. `pitch_tolerance_cents`
stays 9999 (off — enrich handles pitch). No category needs `skip_pitch_shift` (dies with `fx`).

### Prompt authoring → ~150 surviving/category
Need ~200 prompts/category (≈75% prompt-survival → ~150 instruments; multi-source merges pitches into
one instrument, so count tracks *surviving prompts*, not pitch fan-out). **Strategy:** combinatorial
template `[STYLE/ERA] × [TONE] × [PROCESSING] × [ARTICULATION] + noun + suffix`, weighted ~55–60% EDM /
~25% hip-hop-lofi / ~15–20% acoustic-orchestral-world (mirrors `kick.txt:3`). Script the skeleton
(deterministic, holds EDM ratio, prints a style histogram), LLM-polish ~30–40 high-specificity lines
(named gear/genres — SA3 responds strongly), hand-curate 5–10 anchor lines/file. Keep `# --- section ---`
headers. **Dedup:** existing exact/whitespace dedup in `list_to_jsonl*.py` (90–93) + add near-dup
(token-set Jaccard ≥0.85) in the generator. **Articulation must match `min_sustain`/`open_ended`** (no
"short stab" for pads). Finalize wording before the run (edits change content-hash → orphan WAVs).
**Drums:** ~150 prompts × 1 variant, keep all (no drum gate today). A lightweight `gate_drums.py`
(reject clipped/silent/bad-onset) is an optional quality add — recommended but not required.

### Compatibility call-outs
Coordinate new **drum role folder names** with the drum-plugin maintainers (it ignores unknown
folders). New pitched IDs must equal folders. Every new pitched category needs a full
`PitchedCategoryConfig(...)` row or `list_to_jsonl_pitched.py` hard-exits. Retiring `fx` / renaming
`hit` contents ⇒ both packs bump a major version.

---

## Part 4 — Packaging & distribution (one big pack, 150/category)

### Zone format — ship 16-bit WAV, not FLAC (perf decision)
**Finding:** Tracktion's `SamplerPlugin` **CPU-decodes FLAC on the message thread** at load
(`PluginManager.h:404-410` — "a 13-zone piano = 13 synchronous decodes — stalling playback on every
generate/shuffle"). WAV/AIFF are **memory-mapped** (instant). The engine mitigates with an off-thread
FLAC→WAV transcode cache (`PluginManager.cpp:1165-1241`), but that just moves the cost to **first-load
latency** + stores **both** FLAC and decoded WAV on disk (~2×). At 150/category this compounds.
**Decision:** render zones as **16-bit WAV** (mmap-able → engine's fast path, no decode stall, no
transcode cache). Only ~15% larger download than the current 24-bit FLAC (16-bit −33% nearly offsets
losing FLAC compression) but eliminates the penalty; 16-bit is ample for normalized, twice-resampled
zone audio. Keep `sources/` at **24-bit WAV** (2-3 per instrument). `enrich_pitched.py` zone writes →
WAV `subtype="PCM_16"`; manifest `zones[].sample` becomes `zones/060.wav`. **No engine/plugin change**
(engine already prefers WAV; resolver already lists `.wav`). *(Alternative if download size dominates:
keep FLAC but transcode-on-install app-side — more app work, no on-disk saving; not recommended here.)*

### Packs
Keep the **existing two packs** (drum + instrument); **no per-category/tiered split, no app-side marker
change.** Bump both versions. Size: ~28 pitched × 150 ≈ 4,200 instruments ⇒ ~33GB naive; the **zone-span
12→7** (1a) + **16-bit WAV zones** (above) bring the instrument pack to ~**20–24GB**. Drum pack ~2–3GB.
- `scripts/build_pack.py` unchanged structurally; rebuild both, read printed `sizeBytes`/`sha256`.
- `sas-app/src/shared/constants/sample-packs.ts`: bump `expectedVersion`, set new `downloadUrl`,
  paste fresh `sizeBytes`/`sha256` for both packs (the only required app-side change; never overwrite a
  published version — `README-PACKS.md`).
- Update `scripts/README-PACKS.md` size table + WAV-zone note.

---

## Part 5 — RunPod run instructions (single full campaign)
**GPU:** **A100 80GB on-demand** (sweet spot: fits SA3-medium with batch 32–64; ~$0.89–1.89/hr). H100 if
faster generation is worth the premium. 100GB container disk, **no network volume** (MooseFS is slow for
many-small-files), SSH template, CUDA 12.x.
**Setup:** `bash scripts/setup.sh` (venv on `/root/.venv`, torch cu128, `requirements.txt`,
stable-audio-tools from git, verify CUDA). HF cache + outputs on `/workspace`.
**Pitched (generate+gate+report on pod):**
```
SAS_MULTI_SOURCE=1 STAGES=generate,gate,report ./scripts/run_pitched.sh   # add --batch-size 32
```
**Drums:** `./scripts/run_all.sh` (generate + postprocess; `--batch-size 32`).
**Transfer + enrich (local CPU):** rsync `outputs/gated` + `outputs/processed` down; run
`STAGES=enrich ./scripts/run_pitched.sh` (rubberband R3 + `ProcessPoolExecutor`).
**Build + publish:** `build_pack.py --pack drums|instruments --version N`; upload to GCP; update
`sample-packs.ts`. **Terminate the pod immediately after generate+gate.**
**Cost/time:** generation ~1–3h batched; **gate dominates** (~tens of k–~100k+ variants) — budget up to
~a day of pod time, still well under $200 (~$20–45). Dial variant counts down if time matters.

---

## Files to change
**Generator (pitch + batching + report):** `scripts/batch_generate.py` (batched + init_audio),
`scripts/enrich_pitched.py` (multi-source merge + 16-bit WAV zones), `scripts/pitched_category_config.py`
(roots/variants/span), `scripts/gate_pitched.py` (ensemble octave check), **new**
`scripts/pitch_report.py`, `scripts/list_to_jsonl_pitched.py` (+`variants` field),
`scripts/run_pitched.sh` (drop global variants, add `report`), `tests/test_pitch_detection.py`
(octave-rescue), `scripts/repair_instrument_pitch.py` (multi-source guard).
**Taxonomy:** `scripts/category_config.py`, `scripts/pitched_category_config.py`,
`scripts/categories.txt`, `scripts/pitched_categories.txt`, **new** `prompts/<cat>.txt` /
`prompts/pitched/<cat>.txt` (+ a prompt-generator script), remove spurious `prompts/plucks.txt`.
**Packaging:** `scripts/build_pack.py` (rebuild), `scripts/README-PACKS.md`,
`sas-app/src/shared/constants/sample-packs.ts` (version/url/size/sha for both packs).
**No changes:** `manifest-types.ts`, `instrument-resolver.ts`, `plugin-host-mixins/instrument.ts`,
`sas-audio-engine/*` (sampler untouched).

---

## Part 6 — Verification (end-to-end)
**Local (CPU, no GPU):** `python tests/test_pitch_detection.py` (incl. new octave-rescue);
`build_pack.py --smoke-test` extended with a **multi-source manifest fixture** (2 sources, merged
disjoint zones) asserting resolver invariants; dry-inspect `list_to_jsonl_pitched.py` multi-source
fan-out (rows = prompts × pitches, `variants` present); hand-fixture enrich slice → assert one merged
instrument, `len(sources)>=2`, disjoint ordered zones.
**Pod slice before the full campaign (pre-flight, not a wave):** ~20 basses prompts,
`SAS_MULTI_SOURCE=1 --batch-size 16`, `generate,gate,report`: verify batched audio sanity vs a
batch_size=1 spot-check, skip-existing no-ops on rerun, report emits cents histogram + pitch-class
accuracy. Run the 1d init_audio A/B here. Then enrich the slice and validate manifests against
`manifest-types.ts`; `build_pack.py` the slice to extrapolate full size.
**Acceptance gates before committing the full run:** (1) pitch-class accuracy on basses+synths slice
**≥80%**; (2) all-variants-fail rate not worse than current; (3) extrapolated instrument pack ≤ ~24GB.
**After full run:** diff `pitch_summary.md` vs baseline; build both packs; load in `sas-app` and confirm
the instrument plugin resolves multi-source manifests and the drum plugin sees new role folders.

---

## Risks & ordering
**Order:** 1e report + 1c ensemble (CPU, enables measurement) → 1a multi-source enrich → 1b variants +
JSONL `variants` → 2 batching (before scaling prompts, or GPU bill balloons) → 3 prompt authoring →
1d init_audio A/B (independent) → 4 packaging last (after real counts/sizes known).
**Risks:** (1) **Gate wall-clock** is the main schedule risk — mitigate with the GPU pre-gate, sub-bass-
only ensemble, parallelism. (2) **Disjoint-zone invariant** is load-bearing (overlaps double-trigger) —
reuse proven midpoint split + pre-write assertion + smoke fixture. (3) **init_audio can collapse
variety** — A/B must check survival, default off. (4) **Content-hash churn** — freeze prompt wording
before GPU spend. (5) **Drum role coordination** with the plugin. (6) `repair_instrument_pitch.py`
multi-source guard to avoid silent corruption.
