"""
Quick qualitative/quantitative check of Dataset/cricket_synth_best_model.pt
on held-out cricket-synth validation samples. Rebuilds the exact same
clip-based train/val split used in train_cricket_synth.py (same seed=42),
picks a handful of validation triplets, runs inference, and saves a grid
image comparing predicted vs. ground-truth ball position for visual
inspection.

Usage: .venv/Scripts/python.exe check_predictions.py
Output: Dataset/cricket_synth_predictions_preview.png
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import resume_training as rt
import train_cricket_synth as tcs

NUM_SAMPLES = 6
MODEL_PATH = "Dataset/cricket_synth_best_model.pt"
OUT_PATH = "Dataset/cricket_synth_predictions_preview.png"
DETECTION_THRESHOLD = 0.5


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    print("Building samples...")
    samples = tcs.build_samples_cricket_synth()
    print(f"Total triplets: {len(samples)}")

    dataset = rt.TrackNetDataset(samples)
    heatmap_dataset = rt.TrackNetHeatmapDataset(dataset)

    def get_clip_id(sample):
        return os.path.normpath(os.path.dirname(sample[0]))

    clip_ids = [get_clip_id(s) for s in samples]
    unique_clips = sorted(set(clip_ids))

    rng = random.Random(42)
    shuffled_clips = unique_clips[:]
    rng.shuffle(shuffled_clips)

    n_val_clips = max(1, int(len(shuffled_clips) * tcs.VAL_FRACTION))
    val_clips = set(shuffled_clips[:n_val_clips])
    val_indices = [i for i, cid in enumerate(clip_ids) if cid in val_clips]
    print(f"Val samples: {len(val_indices)} ({len(val_clips)} clips)")

    pick_rng = random.Random(7)
    chosen = pick_rng.sample(val_indices, min(NUM_SAMPLES, len(val_indices)))

    model = rt.TrackNetV2().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    errors = []

    with torch.no_grad():
        for row, idx in enumerate(chosen):
            images, labels = dataset[idx]
            heatmap_images, _ = heatmap_dataset[idx]
            logits = model(heatmap_images.unsqueeze(0).to(DEVICE))
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            images_np = images.numpy()

            for frame_idx in range(3):
                ax = axes[row, frame_idx]
                frame = images_np[frame_idx * 3:(frame_idx + 1) * 3].transpose(1, 2, 0)
                ax.imshow(np.clip(frame, 0, 1))

                visibility, gt_x, gt_y = labels[frame_idx].tolist()
                prob_map = probs[frame_idx]
                pmax = prob_map.max()

                if visibility:
                    ax.scatter([gt_x], [gt_y], c="lime", marker="o", s=80,
                               facecolors="none", linewidths=2, label="GT")

                title = f"frame {frame_idx} "
                if pmax > DETECTION_THRESHOLD:
                    py, px = np.unravel_index(np.argmax(prob_map), prob_map.shape)
                    ax.scatter([px], [py], c="red", marker="x", s=80, linewidths=2, label="pred")
                    title += f"(conf={pmax:.2f})"
                    if visibility:
                        dist = ((px - gt_x) ** 2 + (py - gt_y) ** 2) ** 0.5
                        title += f" err={dist:.1f}px"
                        errors.append(dist)
                else:
                    title += f"(no det, max={pmax:.2f})"
                    if visibility:
                        title += " [MISS]"

                ax.set_title(title, fontsize=9)
                ax.axis("off")
                if row == 0 and frame_idx == 0:
                    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=110)
    print(f"Saved preview to {OUT_PATH}")

    if errors:
        errors = np.array(errors)
        print(f"Detections with GT visible: {len(errors)}")
        print(f"Mean pixel error: {errors.mean():.2f}  Median: {np.median(errors):.2f}  Max: {errors.max():.2f}")
    else:
        print("No confident detections on visible-ball frames in this sample.")


if __name__ == "__main__":
    main()
