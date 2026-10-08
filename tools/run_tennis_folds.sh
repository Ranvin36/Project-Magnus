#!/usr/bin/env bash
# Out-of-fold tennis detectors (docs/design/gnn_tracker_selection.md): for each fold,
# train with exactly tennis_v1's settings, pick the epoch on game 8 with the same rule,
# and extract candidates for the fold's held-out games. Games 9-10 are never used.
#
# Run on a GPU pod from the repo root, after the tennis cache is in data/tennis/512x288:
#   nohup bash tools/run_tennis_folds.sh > folds.log 2>&1 &
# Produces notebooks/tennis_folds.tar.gz (candidates + selections + logs, ~15 MB).
set -euo pipefail

python data_prep/make_tennis_folds.py

for F in A B C; do
  RUN="tennis_fold$F"
  echo "=== fold $F: training ($(date +%H:%M)) ==="
  env DATASET=tennis SPLITS_FILE="splits_fold$F.json" RUN_NAME="$RUN" BATCH_SIZE=16 NUM_EPOCHS=20 \
      EARLY_STOPPING_PATIENCE=4 WARMUP_STEPS=500 CHECKPOINT_EVERY=500 \
      python tools/run_notebook.py --train-only > "$RUN.log" 2>&1
  grep -E "epoch .* done|val detection|early stopping|TRAINING STOPPED" "$RUN.log" || true

  echo "=== fold $F: selecting epoch on game 8 ($(date +%H:%M)) ==="
  python -m tracking.select_epoch --run "$RUN" --splits-file "splits_fold$F.json" | tee -a "$RUN.log"
  SEL="checkpoints/$RUN/${RUN}_selection.json"
  EP=$(python -c "import json; print(json.load(open('$SEL'))['selected']['epoch'])")
  TH=$(python -c "import json; print(json.load(open('$SEL'))['selected']['vis_threshold'])")

  echo "=== fold $F: candidates for held-out games (epoch $EP @ $TH) ($(date +%H:%M)) ==="
  python -m tracking.extract_candidates --run "$RUN" --epoch "$EP" --vis-threshold "$TH" \
      --splits-file "splits_fold$F.json" test | tee -a "$RUN.log"
done

tar -czf notebooks/tennis_folds.tar.gz \
    data/tennis/derived/candidates_tennis_fold*_test.json \
    checkpoints/tennis_fold*/*_selection.json checkpoints/tennis_fold*/*_summary.json \
    checkpoints/tennis_fold*/*_training_log.txt tennis_fold*.log
echo "=== done ($(date +%H:%M)): notebooks/tennis_folds.tar.gz ==="
ls -lh notebooks/tennis_folds.tar.gz
