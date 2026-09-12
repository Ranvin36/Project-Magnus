"""
Train the TrackNetV2-architecture ball-detection model from scratch on the
cricket-synth dataset (no TrackNetV2/shuttlecock pretraining -- a prior
fine-tune attempt was scrapped because of a label bug, see below).

Reuses the model/dataset classes from resume_training.py unchanged -- only
the sample-building step differs, since cricket-synth stores per-frame ball
position in JSON rather than TrackNetV2's CSV (Visibility/X/Y) format.

cricket-synth layout (see D:\\Git-Repos\\MAGNUS\\Ball-Tracking\\cricket-synth\\out):
  clips/<clip_id>/f_%04d.jpg   -- frames
  labels/<clip_id>.json        -- one entry per frame: visible, centre_px
                                   (ABSOLUTE pixel coords, top-left origin,
                                   +v downward -- verified against all
                                   45,733 visible=true frames in the
                                   dataset: centre_px lands within
                                   [0,width]x[0,height] 100% of the time
                                   used directly, vs. only ~20% of the time
                                   if camera.intrinsics.cx/cy is added to
                                   it. An earlier version of this script
                                   added the cx/cy offset -- that was wrong
                                   and produced a fine-tuned checkpoint
                                   trained on garbage coordinates, which was
                                   discarded.)

Usage: .venv/Scripts/python.exe train_cricket_synth.py
"""
import copy
import json
import os
import random
import time

import torch
import torch.nn as nn

import resume_training as rt

CRICKET_SYNTH_ROOT = r"D:\Git-Repos\MAGNUS\Ball-Tracking\cricket-synth\out"

WINDOW_SIZE = rt.WINDOW_SIZE
VAL_FRACTION = rt.VAL_FRACTION
POS_WEIGHT = rt.POS_WEIGHT

BATCH_SIZE = 4  # local RTX 2050 has 4GB VRAM
LR = 1e-3  # from-scratch training rate (matches kaggle_train.py / resume_training.py)
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
CHECKPOINT_EVERY = 200

CHECKPOINT_PATH = "Dataset/cricket_synth_checkpoint.pt"
LOG_PATH = "Dataset/cricket_synth_training_log.txt"
BEST_MODEL_PATH = "Dataset/cricket_synth_best_model.pt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def build_samples_cricket_synth(root=CRICKET_SYNTH_ROOT):
    clips_dir = os.path.join(root, "clips")
    labels_dir = os.path.join(root, "labels")

    samples = []
    n_clips_used = 0
    n_clips_skipped = 0

    for label_file in sorted(os.listdir(labels_dir)):
        if not label_file.endswith(".json"):
            continue
        clip_id = label_file[:-5]
        clip_dir = os.path.join(clips_dir, clip_id)
        if not os.path.isdir(clip_dir):
            n_clips_skipped += 1
            continue

        with open(os.path.join(labels_dir, label_file)) as f:
            meta = json.load(f)

        frame_pattern = meta["frame_pattern"]

        frame_paths = []
        frame_labels = []
        ok = True
        for entry in meta["frames"]:
            frame_path = os.path.join(clip_dir, frame_pattern % entry["frame"])
            if not os.path.exists(frame_path):
                ok = False
                break
            visible = bool(entry.get("visible")) and entry.get("centre_px") is not None
            if visible:
                # centre_px is already an absolute top-left-origin pixel coordinate.
                x, y = entry["centre_px"]
            else:
                x, y = 0.0, 0.0
            frame_paths.append(frame_path)
            frame_labels.append((1.0 if visible else 0.0, x, y))

        if not ok or len(frame_paths) < WINDOW_SIZE:
            n_clips_skipped += 1
            continue

        for i in range(len(frame_paths) - WINDOW_SIZE + 1):
            samples.append((
                frame_paths[i], frame_paths[i + 1], frame_paths[i + 2],
                frame_labels[i], frame_labels[i + 1], frame_labels[i + 2],
            ))
        n_clips_used += 1

    log(f"cricket-synth: {n_clips_used} clips used, {n_clips_skipped} skipped (missing frames/dir)")
    return samples


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    log("Building samples from cricket-synth...")
    samples = build_samples_cricket_synth()
    log(f"Built {len(samples)} sliding-window triplets")

    dataset = rt.TrackNetDataset(samples)
    heatmap_dataset = rt.TrackNetHeatmapDataset(dataset)

    def get_clip_id(sample):
        return os.path.normpath(os.path.dirname(sample[0]))

    clip_ids = [get_clip_id(s) for s in samples]
    unique_clips = sorted(set(clip_ids))

    rng = random.Random(42)
    shuffled_clips = unique_clips[:]
    rng.shuffle(shuffled_clips)

    n_val_clips = max(1, int(len(shuffled_clips) * VAL_FRACTION))
    val_clips = set(shuffled_clips[:n_val_clips])
    train_clips = set(shuffled_clips[n_val_clips:])

    train_indices = [i for i, cid in enumerate(clip_ids) if cid in train_clips]
    val_indices = [i for i, cid in enumerate(clip_ids) if cid in val_clips]
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    log(f"Train samples: {len(train_indices)} ({len(train_clips)} clips), "
        f"Val samples: {len(val_indices)} ({len(val_clips)} clips)")

    train_subset_full = torch.utils.data.Subset(heatmap_dataset, train_indices)
    val_subset_full = torch.utils.data.Subset(heatmap_dataset, val_indices)

    full_train_loader = torch.utils.data.DataLoader(
        train_subset_full, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True,
    )
    full_val_loader = torch.utils.data.DataLoader(
        val_subset_full, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=True,
    )

    model = rt.TrackNetV2().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    USE_AMP = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    start_epoch = 0
    step_in_epoch = 0
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    running_loss = 0.0
    running_count = 0

    if os.path.exists(CHECKPOINT_PATH):
        log(f"Found checkpoint at {CHECKPOINT_PATH}, resuming...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"]
        step_in_epoch = ckpt["step_in_epoch"]
        history = ckpt["history"]
        best_val_loss = ckpt["best_val_loss"]
        best_state = ckpt["best_state"]
        patience_counter = ckpt["patience_counter"]
        running_loss = ckpt["running_loss"]
        running_count = ckpt["running_count"]
        log(f"Resumed at epoch {start_epoch + 1}, step {step_in_epoch}, "
            f"running_train_loss={running_loss / max(running_count, 1):.4f}")
    else:
        log("No checkpoint found, starting fresh from random init.")

    steps_per_epoch = len(full_train_loader)

    def save_checkpoint(epoch, step):
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "step_in_epoch": step,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "patience_counter": patience_counter,
            "running_loss": running_loss,
            "running_count": running_count,
        }, CHECKPOINT_PATH)

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        resume_step = step_in_epoch if epoch == start_epoch else 0
        if epoch != start_epoch:
            running_loss = 0.0
            running_count = 0

        for step, (images, targets) in enumerate(full_train_loader):
            if step < resume_step:
                continue

            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            running_count += images.size(0)

            current_step = step + 1
            if current_step % CHECKPOINT_EVERY == 0:
                save_checkpoint(epoch, current_step)
                log(f"  epoch {epoch + 1} step {current_step}/{steps_per_epoch} "
                    f"running_train_loss={running_loss / running_count:.4f} (checkpoint saved)")

        train_loss = running_loss / running_count

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, targets in full_val_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_running_loss += loss.item() * images.size(0)
        val_loss = val_running_loss / len(val_subset_full)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed_min = (time.time() - epoch_start) / 60
        current_lr = optimizer.param_groups[0]['lr']
        log(f"epoch {epoch + 1}/{NUM_EPOCHS} done - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"lr: {current_lr:.2e}  ({elapsed_min:.1f} min)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, BEST_MODEL_PATH)
            log(f"New best model saved (val_loss={best_val_loss:.4f}) -> {BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter > EARLY_STOPPING_PATIENCE:
                log(f"Early stopping after {epoch + 1} epochs - no val_loss improvement for "
                    f"{EARLY_STOPPING_PATIENCE} consecutive epochs.")
                save_checkpoint(epoch + 1, 0)
                break

        step_in_epoch = 0
        save_checkpoint(epoch + 1, 0)

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"Restored best model weights (val_loss={best_val_loss:.4f}).")

    log("Training complete.")


if __name__ == "__main__":
    main()
