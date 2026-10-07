"""
Run notebooks/ball_detection_tracking.ipynb headless (RunPod, Kaggle, any
Linux box), straight from the notebook's own cells -- the notebook stays the
single source of truth, this only executes it.

Settings come from environment variables (see the notebook's Run
Configuration cell), e.g.:

  RUN_NAME=bs16_lr3e-4 BATCH_SIZE=16 LR=3e-4 python tools/run_notebook.py

Modes:
  (default)          train, then evaluate on the held-out test set
  --train-only       stop right after training -- use this for hyperparameter
                     probes, so tuning decisions never look at the test set
  --with-videos      also run the real-footage tracking cells at the end
                     (needs data/cricket-synth/videos/*.mp4, which isn't in git)

Stdout shows each cell's prints as they happen; run it inside tmux (or with
nohup) so it survives a dropped SSH/browser connection.
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # no display on a server; figures are created then discarded
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, "notebooks", "ball_detection_tracking.ipynb")
TRAINING_CELL_MARKER = "class TrainingDiverged"
FIRST_VIDEO_CELL_MARKER = "def read_video_frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default=NOTEBOOK)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--with-videos", action="store_true")
    args = parser.parse_args()

    with open(args.notebook, encoding="utf-8") as f:
        code_cells = ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]

    # The notebook uses paths relative to its own folder (../checkpoints etc.).
    os.chdir(os.path.dirname(os.path.abspath(args.notebook)))
    plt.show = lambda *a, **k: plt.close("all")

    namespace = {"__name__": "__main__"}
    start = time.time()
    for i, source in enumerate(code_cells):
        if FIRST_VIDEO_CELL_MARKER in source and not args.with_videos:
            print("\n[run_notebook] skipping the real-footage video cells (pass --with-videos to run them)")
            break
        first_line = source.strip().splitlines()[0] if source.strip() else ""
        print(f"\n[run_notebook] cell {i + 1}/{len(code_cells)}: {first_line[:80]}", flush=True)
        exec(compile(source, f"<notebook cell {i + 1}>", "exec"), namespace)
        sys.stdout.flush()
        if TRAINING_CELL_MARKER in source and args.train_only:
            print("\n[run_notebook] --train-only: stopping after the training cell")
            break
    print(f"\n[run_notebook] done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
