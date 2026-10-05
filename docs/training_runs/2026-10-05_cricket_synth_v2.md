# Training run: `cricket_synth_v2` — 2026-10-05

First training run of the fixed TrackNetV2 model (full-resolution skip + visibility head), trained on RunPod.

## Headline results — held-out test set

178 test clips, 34,218 frames. Never used for training or checkpoint selection.

| Metric | v1 (old model) | **v2 (this run)** |
|---|---|---|
| Ball detected when visible | 99.5% (20,029 / 20,134) | **98.3%** (19,783 / 20,134) |
| Missed when visible | 0.5% (105) | 1.7% (351) |
| False positives on not-visible frames | 41.6% (5,861 / 14,084) | **2.9%** (406 / 14,084) |
| Pixel error @ 512x288 — mean | 9.74 | **2.10** |
| Pixel error @ 512x288 — median | 7.69 | **1.25** |
| Pixel error @ 512x288 — p90 | 13.52 | **3.82** |
| Pixel error @ 512x288 — max | 332.26 | 205.63 |
| Pixel error @ native resolution — mean | not measured | **7.82** |
| Pixel error @ native resolution — median | not measured | **4.45** |
| Pixel error @ native resolution — p90 | not measured | **14.48** |
| Pixel error @ native resolution — max | not measured | **771.10** |

Detection threshold: v1 = heatmap peak > 0.5; v2 = visibility probability > 0.5 (`VIS_THRESHOLD`). Pixel errors are measured on detected frames where the ball is visible.

For scale: the median ball radius in the dataset is 4.07 px at native resolution, so the v2 median error (4.45 px) is about one ball radius.

## Model and code

- **Code:** commit `290a9a0` ("add: fix bugs"), `notebooks/ball_detection_tracking.ipynb`.
- **Architecture changes vs v1:**
  - **Full-resolution skip.** `e1` is now concatenated after `up4` and refined by `dec4_conv`. In v1 it was computed but never used.
  - **Visibility head.** Bottleneck global max + avg pool → MLP → per-frame visibility logit. Detection requires visibility probability > 0.5; the heatmap gives only the position.
  - **11.6M parameters.**
- **Training-stability fixes vs v1:**
  - v1 diverged to NaN in epoch 2: all weights were NaN and the AMP grad scale had collapsed to 0.
  - LR 1e-3 → 3e-4, with linear warmup.
  - Gradient-norm clipping at 1.0.
  - Non-finite batches are skipped, and training stops after 20 bad batches in a row.
  - Checkpoints refuse to save non-finite state.
- **Loss:** weighted heatmap BCE (`pos_weight=200`) + 0.1 × visibility BCE.

## Data

- **Source:** cricket-synth, 1,192 labelled clips, from the 512x288 cache (`out_512x288`, built by `training/build_resized_cache.py`, JPEG q95).
- **Cache verification:**
  - Labels identical to the full-res pipeline.
  - Images differ by 0.93/255 per pixel on average (JPEG re-encode).
- **Split:** by clip, seed 42, identical to v1.

| Split | Samples | Clips |
|---|---|---|
| Train | 49,594 | 836 |
| Val | 11,007 | 178 |
| Test | 11,406 | 178 |

## Hardware

- **GPU:** RunPod, NVIDIA GeForce RTX 4090 (24 GB).
- **Measured during training:** 97% GPU utilisation, 8.3 GB VRAM, 330 W.
- **Data loading:** 8 DataLoader workers (Linux).
- **Speed:** 7.7–7.8 min per epoch (3,100 steps at batch 16, plus a validation pass over 11,007 samples).
- **Comparison:** v1 on the local RTX 2050 laptop took ~137–143 min per epoch.
- **Cost:** not recorded here. Check RunPod billing. The pod ran from about 13:00 to after 14:05 (pod time, UTC); the Community Cloud RTX 4090 rate at the time was $0.34/h.

## Configuration

From `cricket_synth_v2_summary.json`:

| Setting | Value |
|---|---|
| RUN_NAME | cricket_synth_v2 |
| BATCH_SIZE | 16 |
| LR | 3e-4 (peak; ReduceLROnPlateau factor 0.5, patience 1) |
| WARMUP_STEPS | 1000 |
| NUM_EPOCHS | 10 (ceiling) |
| EARLY_STOPPING_PATIENCE | 2 |
| POS_WEIGHT | 200.0 |
| VIS_LOSS_WEIGHT | 0.1 |
| GRAD_CLIP_NORM | 1.0 |
| CHECKPOINT_EVERY | 2000 |
| NUM_WORKERS | 8 |
| AMP | on |
| HEATMAP_SIGMA | 5 |

## Per-epoch results — validation set

Val metrics are at 512x288. The GT pixel is the argmax of the target heatmap, so errors are quantised to about 0.5 px.

| Epoch | LR | Train loss | Val loss | Val heatmap | Val vis | Detected | False pos. | Mean err (px) | Saved as best |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3.0e-4 | 0.1627 | 0.0463 | 0.0353 | 0.1106 | 97.8% | 8.0% | 4.89 | yes |
| 2 | 3.0e-4 | 0.0343 | 0.0369 | 0.0270 | 0.0998 | 98.3% | 7.0% | 2.91 | yes |
| 3 | 3.0e-4 | 0.0280 | 0.0345 | 0.0259 | 0.0863 | 98.4% | 5.8% | 2.73 | yes |
| 4 | 3.0e-4 | 0.0249 | 0.0331 | 0.0247 | 0.0834 | 97.0% | 3.8% | 2.34 | yes |
| 5 | 3.0e-4 | 0.0232 | 0.0351 | 0.0273 | 0.0777 | 97.6% | 3.8% | 2.11 | no |
| 6 | 1.5e-4 | 0.0216 | 0.0341 | 0.0254 | 0.0872 | 99.2% | 7.9% | 2.14 | no — early stop |
| 7* | 1.5e-4 | 0.0211 | **0.0305** | 0.0233 | 0.0712 | 98.4% | 4.2% | 2.02 | **yes — final model** |

\* **Epoch 7.** The training cell was re-run without restarting the kernel. Training continued from the restored epoch-4 weights at LR 1.5e-4, and the log labels this epoch "epoch 1/10" because the epoch counter restarted. See *Incidents*.

No steps were skipped (no non-finite losses) in any epoch.

## Final model

- **File:** `checkpoints/cricket_synth_v2_best_model.pt` (46,513,871 bytes, saved 14:04:26 pod time).
- **Validation:** val loss 0.0305; 98.4% detected; 4.2% false positives; 2.02 px mean error at 512x288.
- **Integrity check after download:**
  - 140 tensors, all finite.
  - `vis_head` and `dec4_conv` present.

## Incidents and caveats

1. **Training was unintentionally continued after early stopping.** The training cell was run a second time in the same kernel. It resumed from the restored best weights (epoch 4) at the reduced LR and produced a new best (epoch 7*). That turned out to be useful, but it wasn't planned.
2. **`cricket_synth_v2_summary.json` is stale.** It was written at 13:54 by the first run (best val loss 0.0331, 18,600 steps). It doesn't include epoch 7*. The training log is complete up to 14:04:26.
3. **Which weights the test set evaluated (resolved).** On the pod the test cell evaluated whatever `best_model.pt` held at the time, which left some doubt. Re-running the evaluation locally on the downloaded file reproduced the numbers exactly (see *Local verification*), so the tested model is the downloaded epoch-7* model.
4. **Epochs 4, 5 and 6 weights are lost.**
   - Only `best_model.pt` and the rolling `checkpoint.pt` are kept, and both are overwritten.
   - Epoch 5 had better detection metrics than epoch 4 but was never saved, because selection is by val loss.
5. **Early stopping fired in the same epoch as the first LR reduction** (scheduler patience 1 vs. early-stopping patience 2), so the lower LR had no chance to help until the accidental continuation showed it does.

## Local verification (2026-10-05, RTX 2050)

The downloaded `cricket_synth_v2_best_model.pt` was re-evaluated locally through the notebook, with training skipped.

**Test set:** identical to the pod run.
- 19,783 / 20,134 detected (98.3%).
- 406 / 14,084 false positives (2.9%).
- Error at 512x288: mean / median / p90 = 2.10 / 1.25 / 3.82.
- Error at native resolution: mean / median / p90 = 7.82 / 4.45 / 14.47.
  - p90 differs by 0.01 px from the pod (GPU numeric differences).

This confirms the test numbers above came from this exact file.

**`dataset/test/*.mp4` are not real footage.** They are video encodes of synthetic cricket-synth clips:
- Same frame counts as the clips.
- First-frame mean difference of about 3–4/255, i.e. video compression.
- Their labels exist, and one of the three clips was in the training set:

| Video | Source clip | Split | Visible | Detected | Missed | False pos. | Error @1920 px, median / mean / max |
|---|---|---|---|---|---|---|---|
| `main1000_000009_test.mp4` | `main1000_000009` | **test** | 26 | 26 | 0 | 4 / 20 | 4.0 / 6.1 / 21.7 |
| `main1000_000011_test.mp4` | `main1000_000011` | **train** | 35 | 35 | 0 | 1 / 22 | 4.2 / 8.0 / 34.4 |
| `main1000_000027_test.mp4` | `main1000_000027` | **val** | 54 | 54 | 0 | 0 / 39 | 3.1 / 3.9 / 12.0 |

Pipeline: video tracker with visibility threshold 0.5, persistence filter and trajectory gate.

Implications:
- **The v1 "missed ~35–40% of real frames" reading was wrong.** Those were frames where the ball isn't in view (run-up and pre-roll), not misses.
- **Nothing in the project has yet been tested on real broadcast footage.**

## Known issues and next steps

- **Real footage not yet tested.** The `dataset/test/` videos turned out to be synthetic (see *Local verification*), so performance on real broadcast footage is still unknown. This is the main open question.
- **Large outliers.** Max native error is 771 px, and the mean (7.82) is well above the median (4.45), so some detections lock onto the wrong object. The physics stage needs robust fitting or outlier rejection.
- **Integer-argmax position read-out** limits precision. The median of 1.25 px at 512x288 is close to that floor, so sub-pixel decoding (no retraining) is the next cheap gain.
- **The visibility threshold (0.5) is untuned.** Tune it on the val set to choose the miss / false-positive trade-off deliberately.
- **Checkpoint selection uses val loss**, not detection metrics. Select by error and false-positive rate instead, and keep per-epoch weight snapshots.
- **Patience settings:** early-stopping patience should exceed the scheduler's patience so a reduced LR gets at least one epoch.
- **App not updated.** `server/detection.py` still loads the v1 architecture and can't load these weights yet.

## Artifacts (local, `checkpoints/`, not in git)

| File | Size | Notes |
|---|---|---|
| `cricket_synth_v2_best_model.pt` | 46.5 MB | final model (epoch 7*) |
| `cricket_synth_v2_training_log.txt` | 3.2 KB | complete to 14:04:26 |
| `cricket_synth_v2_summary.json` | 1.8 KB | stale, first run only (see Incidents) |
| `~/Downloads/results_v2.tar.gz` | 43.1 MB | archive the above came from |

## Raw training log

Times are pod time (UTC).

```
[13:07:59] No checkpoint found -- starting fresh from random init on the 70/15/15 split.
[13:12:35]   epoch 1 step 2000/3100 running_train_loss=0.2275 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:15:46] epoch 1/10 done - train_loss: 0.1627  val_loss: 0.0463 (heatmap 0.0353, vis 0.1106)  lr: 3.00e-04  skipped steps: 0  (7.8 min)
[13:15:46]   val detection: 97.8% detected, 8.0% false positives, mean error 4.89px @512x288
[13:15:46] New best model saved (val_loss=0.0463) -> ../checkpoints/cricket_synth_v2_best_model.pt
[13:20:18]   epoch 2 step 2000/3100 running_train_loss=0.0353 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:23:26] epoch 2/10 done - train_loss: 0.0343  val_loss: 0.0369 (heatmap 0.0270, vis 0.0998)  lr: 3.00e-04  skipped steps: 0  (7.7 min)
[13:23:26]   val detection: 98.3% detected, 7.0% false positives, mean error 2.91px @512x288
[13:23:26] New best model saved (val_loss=0.0369) -> ../checkpoints/cricket_synth_v2_best_model.pt
[13:27:58]   epoch 3 step 2000/3100 running_train_loss=0.0285 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:31:07] epoch 3/10 done - train_loss: 0.0280  val_loss: 0.0345 (heatmap 0.0259, vis 0.0863)  lr: 3.00e-04  skipped steps: 0  (7.7 min)
[13:31:07]   val detection: 98.4% detected, 5.8% false positives, mean error 2.73px @512x288
[13:31:07] New best model saved (val_loss=0.0345) -> ../checkpoints/cricket_synth_v2_best_model.pt
[13:35:38]   epoch 4 step 2000/3100 running_train_loss=0.0251 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:38:46] epoch 4/10 done - train_loss: 0.0249  val_loss: 0.0331 (heatmap 0.0247, vis 0.0834)  lr: 3.00e-04  skipped steps: 0  (7.7 min)
[13:38:46]   val detection: 97.0% detected, 3.8% false positives, mean error 2.34px @512x288
[13:38:46] New best model saved (val_loss=0.0331) -> ../checkpoints/cricket_synth_v2_best_model.pt
[13:43:18]   epoch 5 step 2000/3100 running_train_loss=0.0234 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:46:27] epoch 5/10 done - train_loss: 0.0232  val_loss: 0.0351 (heatmap 0.0273, vis 0.0777)  lr: 3.00e-04  skipped steps: 0  (7.7 min)
[13:46:27]   val detection: 97.6% detected, 3.8% false positives, mean error 2.11px @512x288
[13:50:58]   epoch 6 step 2000/3100 running_train_loss=0.0216 lr=3.00e-04 skipped=0 (checkpoint saved)
[13:54:07] epoch 6/10 done - train_loss: 0.0216  val_loss: 0.0341 (heatmap 0.0254, vis 0.0872)  lr: 1.50e-04  skipped steps: 0  (7.7 min)
[13:54:07]   val detection: 99.2% detected, 7.9% false positives, mean error 2.14px @512x288
[13:54:07] early stopping after 6 epochs -- no val_loss improvement for 2 consecutive epochs
[13:54:07] Run summary -> ../checkpoints/cricket_synth_v2_summary.json
[13:54:07] Restored best model weights (val_loss=0.0331).
[13:54:07] Training complete.
[14:01:17]   epoch 1 step 2000/3100 running_train_loss=0.0213 lr=1.50e-04 skipped=0 (checkpoint saved)
[14:04:26] epoch 1/10 done - train_loss: 0.0211  val_loss: 0.0305 (heatmap 0.0233, vis 0.0712)  lr: 1.50e-04  skipped steps: 0  (7.7 min)
[14:04:26]   val detection: 98.4% detected, 4.2% false positives, mean error 2.02px @512x288
[14:04:26] New best model saved (val_loss=0.0305) -> ../checkpoints/cricket_synth_v2_best_model.pt
```

## Raw test-set evaluation output

```
Test set: 11406 triplets from 178 clips, 34218 frames total (never used for training or checkpoint selection)
  visible-ball frames: 20134   not-visible frames: 14084

On visible-ball frames (VIS_THRESHOLD=0.5):
  detected: 19783/20134 (98.3%)
  missed:   351/20134 (1.7%)
  pixel error @ 512x288 input -- mean: 2.10  median: 1.25  p90: 3.82  max: 205.63
  pixel error @ native resolution -- mean: 7.82  median: 4.45  p90: 14.48  max: 771.10

On not-visible frames:
  false positives: 406/14084 (2.9%)
```
