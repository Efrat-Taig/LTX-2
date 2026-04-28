# Phase 0.2 — Quality Triage Report

Source: `docs/phase0/quality_triage.csv` (1439 clips triaged).
GCS videos skipped (bandwidth): **308549** — see follow-up at end.

## Bucket eligibility thresholds (committed in source before run)

| Bucket | Sharpness | Letterbox edge | Motion (mean Δ) |
|---|---|---|---|
| Static | ≥ p25 (109.9) | ≤ 0.05 both edges | < 0.01 |
| Dynamic | ≥ p25 (109.9) | ≤ 0.05 both edges | 0.025–0.15 |
| Closeup | ≥ p25 (109.9) | ≤ 0.05 both edges | any |

p25 floor on sharpness is set empirically from this run's distribution and committed to the next run's report — re-runs use the prior p25 unless the script's constant is bumped.

## Sharpness distribution

| Laplacian variance | Count |
|---|---:|
| 0–25 | 282 |
| 25–50 | 32 |
| 50–100 | 20 |
| 100–200 | 377 |
| 200–500 | 684 |
| 500+ | 19 |

## Motion distribution

| Motion (mean abs frame Δ, [0,1]) | Count |
|---|---:|
| 0 – 0.010 (static-eligible) | 0 |
| 0.010 – 0.025 (between) | 6 |
| 0.025 – 0.150 (dynamic-eligible) | 690 |
| 0.150+ (over-motion) | 718 |

## Bucket eligibility — shortfall vs vision.md targets

Per character (Chase / Skye), counted independently. A clip can be eligible for 0 or 1 buckets here (we pick the strictest match).

| Character | Bucket | Have | Target | Gap |
|---|---|---:|---:|---:|
| chase | static | 0 | 50 | 50 |
| chase | dynamic | 0 | 70 | 70 |
| chase | closeup | 0 | 30 | 30 |
| skye | static | 0 | 50 | 50 |
| skye | dynamic | 10 | 70 | 60 |
| skye | closeup | 0 | 30 | 30 |

## Audio presence

- Clips with audio track: **1413/1439**
- Per-clip mean volume (dBFS) in `quality_triage.csv` column `audio_level_db`. Whisper SNR estimate is deferred (Step 2 work, frozen).

## Skipped metric: watermark detection

Status: **SKIPPED — no templates configured**

Per `active_plan.md §3.2`, watermark detection requires the PP logo template crops. Until templates are provided in `pre_flight.yaml.phase0.pp_watermark_templates`, this metric is omitted. **Followup**: supply 2–3 cropped logo PNGs (the official PP logo as it appears in source content) — bottom-left, top-right corners are typical.

## Followup: GCS videos not triaged

308549 GCS videos were not triaged. Triaging them requires either:

- Streaming each video locally (bandwidth-prohibitive at the 14TB scale of `video_gen_dataset`).
- Running the triage on a GPU VM that already has the data mounted (preferred — saves bandwidth, parallelizes).

Defer to Phase 1 — once the MLOps scaffolding lands, this can be a Cloud Run job over the bucket. For now, focus on the local clips, which represent the curated subset.
