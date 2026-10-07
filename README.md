# Project Magnus

Single-camera ball tracking and physics-informed 3D trajectory estimation, across sports
(cricket, tennis, table tennis).

## Layout

```
notebooks/
  ball_detection_tracking.ipynb   main notebook: data, model, training, evaluation (source of truth)
  archive/                        old notebooks kept for reference
data_prep/                        build datasets into the common format (512x288 cache + labels)
  build_cricket_cache.py          cricket-synth  -> data/cricket-synth/512x288
  convert_tennis.py               TrackNet tennis -> data/tennis/512x288
tools/                            run, compare and benchmark training
  run_notebook.py                 run the notebook headless (RunPod / Kaggle / local)
  compare_runs.py                 one row per run from checkpoints/<run>/*_summary.json
  benchmark_epoch.py              measure epoch time on the current machine
legacy/                           v1-era scripts (old model, known NaN bug) -- do not train with these;
                                  server/ still imports the v1 model from here until it moves to v2
server/                           Flask API (detection + physics pipeline)
frontend/                         React + Vite web app
docs/                             design notes and training-run records

data/          (gitignored)  all datasets, one folder each -- see data/datasets.json
checkpoints/   (gitignored)  one folder per training run (folder name = RUN_NAME)
output/        (gitignored)  generated files only (previews, tracked videos)
```

## Datasets

All datasets live in `data/<name>/512x288/` in the same label format (the cricket-synth schema),
each with a `splits.json`. `data/datasets.json` describes each one: sport, real or synthetic,
resolution, fps, how ball positions are defined and how the split was made.

| Name | Real? | Ground truth | Status |
|---|---|---|---|
| `cricket-synth` | synthetic | 2D + 3D | ready |
| `tennis` | real broadcast | 2D | ready |
| `table-tennis` | real | 3D (multi-camera) | raw only, converter not built yet |

## Common commands

```bash
# train / evaluate (settings via environment variables, see the notebook's Run Configuration cell)
DATASET=tennis RUN_NAME=tennis_v1 python tools/run_notebook.py
python tools/compare_runs.py
```

`RUN_NAME` defaults to `<dataset>_run` (e.g. `tennis_run`), so a run on one dataset never writes into
another's folder. To open an existing run, name it: `RUN_NAME=cricket_synth_v2` loads the v2 cricket model.
The train/val/test split always comes from the dataset's own `splits.json`.

```bash

# rebuild a dataset cache
python data_prep/convert_tennis.py
python data_prep/build_cricket_cache.py
```
