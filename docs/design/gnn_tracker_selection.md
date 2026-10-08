# Why a graph tracker? — evidence behind the GNN decision

*Status: decision taken 2026-10-07, based on the measurements below. Some follow-up
experiments are still running (see "Open items").*

## 1. Summary

We will add a **physics-aware graph tracker** on top of the existing TrackNet-style
detector. It reasons over a short window of frames (about 10) and does two jobs:

1. **choose** the right ball candidate when the detector offers several, and
2. **predict** the ball's position through runs of frames where the detector sees
   nothing, using the motion before and after the gap.

We did not choose this because GNNs are fashionable. We chose it because measurements
on real tennis footage show that the detector's remaining failures **come in runs of
consecutive frames**, which a per-frame model cannot fix and a simple threshold or
smoother cannot fix either. On synthetic cricket, the same measurements showed the
problem barely exists — so the decision was driven by real data, not assumed.

## 2. How the decision was made

### 2.1 Rule fixed before looking at results

Before training on real data, we wrote down when a graph tracker would be justified:

> A graph tracker is justified on a sport if, on its held-out test set:
> - **(1)** misses + distractors make up **at least 5% of visible frames**, and
> - **(2a)** at least 50% of failure frames have the true ball among the detector's
>   top-5 candidates, **or (2b)** failure runs of 3 or more consecutive frames make up a
>   significant share of failures.
>
> Otherwise, the detection contribution should be blur-aware, sub-pixel localisation
> instead.

Fixing the rule first protects against reading the results in favour of the idea we
wanted.

### 2.2 Definitions

All distances are at the network input size, 512x288 pixels, unless marked "native"
(the original video resolution).

| Term | Meaning |
|---|---|
| **Correct** | Ball visible, detected, top guess within 3 px of the label |
| **Imprecise** | Ball visible, detected, top guess 3–10 px from the label |
| **Distractor** | Ball visible, detected, top guess more than 10 px away (locked onto something else) |
| **Miss** | Ball visible, but the visibility head says "no ball" |
| **Failure** | Miss or distractor |
| **Top-K headroom** | Share of failure frames where the true ball is among the detector's K strongest candidates — the most that "pick a different candidate" could ever fix |
| **Failure run** | Consecutive frames that are all failures |

### 2.3 Models and data used as evidence

| | Cricket (synthetic) | Tennis (real broadcast) |
|---|---|---|
| Detector | `cricket_synth_v2` (TrackNetV2-derived + full-res skip + visibility head) | `tennis_v1`, epoch 14, visibility threshold 0.6 |
| How that model was chosen | lowest validation loss | pre-registered rule: best F1 on the validation match (4 px tolerance) |
| Test set | 178 held-out clips, 11,762 frames | games 9–10 (two full matches never seen in training), 3,717 frames |
| Split | by clip | **by match** — no match appears in two splits |

## 3. Evidence

### 3.1 Synthetic cricket: no case for a graph tracker

| Visible frames (6,882) | Share |
|---|---|
| Correct | 91.4% |
| Imprecise | 5.6% |
| Miss | 1.7% |
| Distractor | 1.3% |

- **Criterion (1): 3.0% failures — fails** the 5% bar.
- Top-5 headroom: 47 of 207 failures (about 23%) — low.
- Failure runs: 85 of 127 runs are a single frame — failures are isolated.
- Occlusion: 2.4% of frames, mostly 1–2 frames long.

**Conclusion:** on synthetic data the detector rarely fails, and when it does the
failures are isolated. A graph tracker would have almost nothing to fix. If we had only
used synthetic data, we would (correctly) not have built it.

### 3.2 Real tennis: criteria met

**Frames with a visible ball (3,619):**

| Outcome | Frames | Share |
|---|---|---|
| Correct | 3,135 | 86.6% |
| Imprecise | 200 | 5.5% |
| Miss | 167 | 4.6% |
| Distractor | 117 | 3.2% |

**Against the rule:**

| Criterion | Tennis (real) | Cricket (synthetic) | Result |
|---|---|---|---|
| (1) failures ≥ 5% of visible frames | **7.9%** (284 frames) | 3.0% | **pass** |
| (2a) ≥ 50% of failures have the true ball in the top-5 | 14–25% (see 3.3) | ~23% | fail |
| (2b) failure runs of 3+ frames are a significant share | **64% of failure frames** (181 of 284) are in runs of 3–22 frames | mostly single frames | **pass** |

**Failure-run lengths (tennis test):**

| Run length (frames) | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 14 | 22 |
|---|---|---|---|---|---|---|---|---|---|
| Number of runs | 51 | 26 | 9 | 9 | 6 | 5 | 1 | 2 | 1 |

**Result: (1) and (2b) pass, so the graph tracker is justified on real tennis.**

Note on (2b): the rule said "significant share" without a number. 64% of failure frames
being in runs of 3+ is clearly significant, but future versions of this rule should state
the threshold in advance (e.g. "at least 30%").

### 3.3 Why the tracker must *predict*, not only *choose*

Criterion (2a) failed: the true ball is rarely among the detector's candidates when it
fails. We checked this carefully, because an early version of the measurement was
misleading:

| Candidate extraction | True ball in top-5 (all failures) |
|---|---|
| Local maxima, strength > 0.02 | 19% |
| Local maxima, strength > 0.001, up to top-20 | 21% |
| One candidate per blob (> 0.3), tolerance 3 px | 14% |
| One candidate per blob (> 0.3), tolerance 8 px (≈ 20 px native) | 20% |

Looking at the detector's heatmap value **at the true ball** in failure frames:

| Failure type | Heatmap value at the true ball | Interpretation |
|---|---|---|
| **Miss** (165–180 frames) | median 0.0003; below 0.001 in 62–65% | **The detector is blind there.** No candidate exists to choose. |
| **Distractor** (107–118 frames) | median 0.90–0.93; above 0.1 in 70–77% | The ball *is* seen, but the top guess is elsewhere (see 3.4) |
| Correct frames (reference) | median 0.999 | |

**Conclusion:** the largest failure group is frames where the detector sees nothing at
all, usually in runs. Picking a different candidate cannot help there. The tracker must
**predict the ball's position from the trajectory before and after the gap**. This is
exactly the "sparse, occluded observations" part of the project's research question,
applied to tracking.

### 3.4 Not every "distractor" is another object

Distance from the top guess to the label, in the 118 distractor frames:

| Distance (px, 512x288) | Frames | Likely meaning |
|---|---|---|
| 10–25 | 48 | **Same ball, different point on its motion streak** |
| 25–50 | 12 | unclear |
| 50+ | 58 | a genuinely different object |

The tennis labels mark the **leading end** of a blurred ball's streak (stated in the
dataset README). A fast ball travels about 1.9 m between frames at 30 fps, so the
streak can be tens of pixels long. The model's peak often sits elsewhere on the same
streak. So:

- **genuine wrong-object distractors are ~58 frames (~1.6% of visible frames)**, and
- **streak-offset errors are ~50 frames (~1.4%)** — a labelling-convention and
  blur-localisation issue, not a tracking issue.

The 10–25 px interpretation is inferred from the numbers and should be confirmed by
viewing example frames.

### 3.5 Where failures concentrate

| Tennis visibility class | Frames | Correct | Miss | Distractor |
|---|---|---|---|---|
| Easy (class 1) | 3,342 | 89.6% | 3.1% | 2.4% |
| Hard to see (class 2: blurred / faint) | 277 | **50.9%** | **22.7%** | **13.4%** |

Hard-to-see balls are 7.7% of visible frames but produce about **35% of all failures**.
Occlusion is not the driver in tennis: the test matches contain only **6 occluded frames**.

### 3.6 False positives (detector fires when there is no ball)

On the tennis test set the detector fires on **21.4%** of frames with no visible ball
(epoch 14, threshold 0.6). Investigation so far:

| Possible cause | Evidence | Verdict |
|---|---|---|
| Bug in training targets | 200 "no ball" + 200 "ball" frames checked: all targets correct | ruled out |
| Rare "no ball" examples + low loss weight | only 4.0% of training frames have no ball; two loss fixes failed (3.7) | not fixable by reweighting |
| 3-frame input window still shows the ball | false-positive rate 40% one frame after the ball exits vs 17% six or more frames away | partial cause |
| Visibility head too coarse (pooled /16 bottleneck) | still 17% false positives far from any ball | likely part of it |

### 3.7 Two attempts to fix false positives in the detector — both failed

Each run changed exactly one setting from `tennis_v1`. All were compared with the same
rule, fixed before scoring: *for each run, the (epoch, threshold) with the lowest
validation false-positive rate among those detecting at least 95% of visible balls;
thresholds swept 0.05–0.999.*

| Run | Change | Best setting under the rule | Val false positives | Val F1 | Val p90 error | Val >10 px |
|---|---|---|---|---|---|---|
| **`tennis_v1`** | baseline | epoch 8 @ 0.6 | **16.2%** | **0.890** | **3.67 px** | **2.7%** |
| `tennis_v2` | visibility loss weight 0.1 → 0.5 | — (false positives 91–100% in every logged epoch through epoch 10; not scored further) | — | — | — | — |
| `tennis_v3` | "no ball" frames weighted 20x inside the visibility loss | epoch 5 @ 0.2 | 51.4% | 0.736 | 8.77 px | 8.1% |

- **v2:** scaling the whole visibility loss did not lower false positives and made
  localisation worse (the shared network shifted capacity away from finding the ball).
- **v3:** rebalancing made the visibility head react (false positives dropped in some
  epochs) but it was unstable (26–100% across epochs), and to reach 95% detection it
  needed a threshold of 0.2, giving 3x v1's false positives and worse localisation.

**Conclusion:** a single-frame detector cannot reliably decide "is there a ball?" in
tennis. Reweighting the loss does not fix it. Together with the timing evidence above
(false positives peak right after the ball leaves the frame), this makes the decision
**partly temporal**, so it moves into the tracker (section 5, requirement 2).

### 3.8 Frozen detector for all tracker work

**`tennis_v1`, epoch 14, visibility threshold 0.6** — chosen by the original
pre-registered F1 rule (validation F1 0.916) and already tested once on games 9–10
(F1 0.911, 21.4% false positives, median error 3.16 px native). Epoch 8 scores better
on the later false-positive rule, but switching to it after the fact would be selecting
on a rule written later; since the tracker now owns the presence decision, the
detector's own false-positive rate matters less. All baselines and the graph tracker
use this detector's outputs unchanged.

This matters for the tracker in two ways:

1. Part of "is there a ball?" is **temporal** (the ball just left the frame vs. it is
   briefly blurred) — another job a model that sees several frames does naturally.
2. The comparison must be fair: obvious detector fixes are tried first, so the tracker
   is credited only for what genuinely needs temporal reasoning.

## 4. Alternatives considered and rejected

| Alternative | Why not |
|---|---|
| Replace the backbone with HRNet | Not novel (HRNet-based ball detectors are published, e.g. WASB); does not address failures that come in runs; kept only as a fallback if real-footage localisation is the bottleneck |
| ViT / Swin transformer backbone | Patch tokenisation (16x16 or 4x4) loses a 1–2 px ball; data- and compute-hungry; does not address temporal failures |
| Candidate-selection-only GNN | Top-5 headroom is only 14–25%; it cannot recover blind frames, which are the largest failure group |
| Tuning the visibility threshold | Swept 0.5–0.999 on validation: false positives stayed at 43–60% for the later epochs |
| Bigger visibility loss weight | Tried in `tennis_v2`: no improvement, worse localisation (3.7) |
| Rebalancing "no ball" frames in the loss | Tried in `tennis_v3`: unstable, 3x the false positives at matched detection (3.7) |

## 5. What the tracker must do (requirements derived from the evidence)

1. Work over a sliding window of about 10 frames.
2. Use the detector's candidates (one per heatmap blob, with score and sharpness) **plus
   a "no ball" option per frame**.
3. **Predict positions through blind runs** of up to ~20 frames, not only choose among
   candidates.
4. Use **physics-aware edges**: implied velocity, acceleration consistent with gravity,
   bounce-like direction changes, gap length.
5. Output, per frame: position, visible flag, and an uncertainty value for the physics
   stage.
6. Be **sport-agnostic**: same architecture for tennis, table tennis and cricket; only
   training data and physics constants differ.

## 6. How it will be judged

The tracker is compared on the **same detector outputs** against:

| Baseline | What it represents |
|---|---|
| Per-frame decoding (current) | no temporal reasoning |
| Viterbi / dynamic programming over candidates with a physics cost | classical candidate selection |
| Kalman filter with a gravity model, coasting through gaps | classical prediction through gaps |
| TrackNetV3-style learned rectification (if feasible) | the published learned alternative |

It counts as a success if, on the held-out tennis matches (and later table tennis), it:

- recovers more frames inside detector-blind runs than the best baseline,
- has lower position error inside those runs,
- reduces wrong-object frames,
- and does not increase false positives.

If a physics Kalman filter matches it, that is still a valid, reported finding — but a
weaker headline.

## 6b. Prototype results so far (2026-10-08)

All on the frozen detector's candidates; settings chosen on validation (game 8).
**Games 9–10 have now been inspected during development and must be reported as a dev
set; the final claim needs untouched data (e.g. table tennis `game_01`).**

| Method (dev = games 9–10) | F1 | False pos. | Wrong obj. | Failure frames recovered |
|---|---|---|---|---|
| Per-frame detector | **0.912** | 21.4% | 3.2% | 0% |
| Viterbi | 0.897 | **12.2%** | **1.9%** | 10.5% |
| Viterbi + interpolation | 0.897 | 19.4% | 2.7% | 30.4% |
| Kalman (const. accel. + RTS smoothing) | 0.888 | 34.7% | 3.1% | **40.2%** |
| Graph tracker v1 (window 10, 1 seed) | 0.904 | 21.4% | 3.9% | 14.3% |
| Graph tracker v2 (window 32, dilated links, long-gap training; 3 seeds) | 0.900 ± 0.003 | 31.6% ± 9.1 | 4.5% ± 0.6 | 17.4% ± 2.6 |

On validation the graph tracker was the best method (v2: F1 0.936 ± 0.003, 46.6%
recovered), so the gap is generalisation between matches, not capacity: all of its
training failures were transplanted from one match (game 8), the same match used to
select it. Kalman uses no training data, only physics, and generalises better.

Training data: real trajectories from games 1–7 with the frozen detector's real
behaviour on game 8 transplanted onto them (`tracking/simulate.py`); checked to match
game 8's statistics (F1, false positives, wrong-object rate, run lengths). No extra
detectors are trained (out-of-fold detectors were rejected as not scalable across
sports).

**Next (v3): physics-guided graph tracker.** Add the Kalman/RTS predicted position as an
extra, flagged candidate node per frame, so the network only has to learn *which source
to trust* (detector candidates, physics prediction, or none) rather than how to predict
trajectories. If v3 does not beat the baselines, the reported finding becomes
"learned graph tracking overfits to the training match; physics-based tracking
generalises better".

## 7. Caveats

- **One detector, two test matches.** The pattern must be confirmed with the fixed
  detector and on table tennis.
- **Small "no ball" samples:** 98 such frames in the tennis test set, 105 in validation.
  False-positive percentages are noisy.
- **Label conventions differ by sport** (tennis: streak leading end; cricket: ball
  centre). A unified tracker needs one convention.
- **Novelty is not yet confirmed.** Graph networks for multi-object tracking and learned
  trajectory completion exist; a physics-aware graph tracker for single-ball trajectories
  across sports must be checked against the literature before it is claimed as new.

## 8. Open items

- [x] `tennis_v2` (visibility loss weight 0.5) — no improvement (3.7).
- [x] `tennis_v3` ("no ball" frames weighted 20x) — no improvement (3.7).
- [x] Freeze the detector — `tennis_v1` epoch 14 @ 0.6 (3.8).
- [ ] Build candidate extraction and the baselines (per-frame, Viterbi, Kalman) before the tracker.
- [ ] Confirm the streak-offset interpretation (3.4) by viewing example frames.
- [ ] Move the diagnostic scripts into the repository (`tools/`) so every number here can
      be regenerated.
- [ ] Repeat the diagnostic on table tennis once its converter exists.

## 9. How the numbers were produced

| Measurement | Method |
|---|---|
| Model selection (tennis) | Every epoch snapshot scored on the validation match; best F1 (4 px tolerance) chosen; evaluated once on test |
| Failure categories, runs, classes | Each test frame scored once (as the middle of its 3-frame window), heatmap argmax vs. label |
| Top-K headroom | Candidates extracted from the heatmap (local maxima, and one-per-blob variants); true ball counted if within tolerance |
| Heatmap value at the true ball | Maximum heatmap value within ±2 px of the label |
| False-positive timing | Distance in frames from each "no ball" frame to the nearest frame with a visible ball |

The scripts currently live outside the repository (see open items); they read the
notebook's own model code, the `data/tennis/512x288` cache and the run snapshots in
`checkpoints/tennis_v1/`.
