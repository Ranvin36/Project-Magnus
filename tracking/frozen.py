"""
The frozen detector every tracker is built and evaluated on.

Chosen in docs/design/gnn_tracker_selection.md (section 3.8): tennis_v1, epoch 14,
visibility threshold 0.6. All baselines and the graph tracker consume this
detector's outputs unchanged, so the comparison between trackers is fair.
"""
import json
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO, "notebooks", "ball_detection_tracking.ipynb")

DATASET = "tennis"
RUN = "tennis_v1"
EPOCH = 14
VIS_THRESHOLD = 0.6
WEIGHTS = os.path.join(REPO, "checkpoints", RUN, f"{RUN}_epoch{EPOCH:02d}.pt")
TAG = f"{RUN}_ep{EPOCH:02d}"  # names derived outputs (e.g. candidate files)


def weights_path(run=RUN, epoch=EPOCH):
    return os.path.join(REPO, "checkpoints", run, f"{run}_epoch{epoch:02d}.pt")


def load_detector(device, run=RUN, epoch=EPOCH):
    """The notebook's TrackNetV2 (single source of the model code) with a run's epoch
    snapshot -- the frozen detector by default, or e.g. an out-of-fold detector."""
    with open(NOTEBOOK, encoding="utf-8") as f:
        cells = ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    g = {}
    for marker in ("import torch.nn as nn", "class TrackNetV2"):
        exec(next(c for c in cells if marker in c), g)
    model = g["TrackNetV2"]().to(device)
    model.load_state_dict(torch.load(weights_path(run, epoch), map_location=device, weights_only=True))
    return model.eval()
