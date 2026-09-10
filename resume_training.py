"""
Standalone resumable training script for TrackNetV2 ball detection.

Mirrors the "Real Training" section of Ball_Detection_Tracking.ipynb
(cells 17-18, 22, 25, 27, 40-41), plus checkpoint save/resume logic that
the notebook's training cell no longer has. Safe to interrupt and rerun:
it picks up from Dataset/training_checkpoint.pt if present.
"""
import copy
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

VIDEOS_DIR = "Dataset/TrackNetV2"
FRAMES_BASE_OUTPUT = "Dataset/TrackNetV2_extracted_frames"
WINDOW_SIZE = 3
HEATMAP_SIGMA = 5
VAL_FRACTION = 0.2
BATCH_SIZE = 8
POS_WEIGHT = 200.0
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
CHECKPOINT_EVERY = 200

CHECKPOINT_PATH = "Dataset/training_checkpoint.pt"
LOG_PATH = "Dataset/training_log.txt"
BEST_MODEL_PATH = "Dataset/best_model.pt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def generate_heatmap(x, y, visibility, H, W, sigma=HEATMAP_SIGMA):
    if not visibility:
        return np.zeros((H, W), dtype=np.float32)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    heatmap = np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class TrackNetDataset(torch.utils.data.Dataset):
    def __init__(self, samples, resize_to=(512, 288)):
        self.samples = samples
        self.resize_to = resize_to

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path_1, frame_path_2, frame_path_3, label_1, label_2, label_3 = self.samples[idx]
        new_w, new_h = self.resize_to
        images = []
        labels = []
        for frame_path, label in zip((frame_path_1, frame_path_2, frame_path_3), (label_1, label_2, label_3)):
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]
            image = cv2.resize(image, self.resize_to, interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255.0
            images.append(image.transpose(2, 0, 1))

            visibility, x, y = label
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            labels.append((visibility, x * scale_x, y * scale_y))

        images = np.concatenate(images, axis=0)
        labels = np.array(labels, dtype=np.float32)
        return torch.from_numpy(images), torch.from_numpy(labels)


class TrackNetHeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.W, self.H = base_dataset.resize_to

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        images, labels = self.base_dataset[idx]
        heatmaps = np.stack([
            generate_heatmap(x=labels[i, 1].item(), y=labels[i, 2].item(),
                              visibility=labels[i, 0].item(), H=self.H, W=self.W)
            for i in range(labels.shape[0])
        ], axis=0)
        return images.float(), torch.from_numpy(heatmaps)


class TrackNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1_conv(x)
        p1 = self.pool1(e1)
        e2 = self.enc2_conv(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3_conv(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4_conv(p3)
        p4 = self.pool4(e4)

        d1 = self.up1(p4)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1_conv(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2_conv(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3_conv(d3)

        d4 = self.up4(d3)
        out = self.final_conv(d4)
        return out


def build_samples():
    samples = []
    for dirpath, dirnames, filenames in os.walk(FRAMES_BASE_OUTPUT):
        if dirnames:
            continue
        frame_files = sorted(f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not frame_files:
            continue

        rel_path = os.path.relpath(dirpath, FRAMES_BASE_OUTPUT)
        parts = rel_path.split(os.sep)
        clip_name = parts[-1]
        csv_parts = ["csv" if p == "video" else p for p in parts[:-1]]
        csv_path = os.path.join(VIDEOS_DIR, *csv_parts, f"{clip_name}_ball.csv")

        if not os.path.exists(csv_path):
            continue

        clip_labels = pd.read_csv(csv_path)
        n = len(clip_labels)
        frame_paths = [os.path.join(dirpath, f) for f in frame_files[:n]]

        for i in range(len(frame_paths) - WINDOW_SIZE + 1):
            frame_path_1, frame_path_2, frame_path_3 = frame_paths[i], frame_paths[i + 1], frame_paths[i + 2]
            label_1 = tuple(clip_labels.iloc[i][["Visibility", "X", "Y"]])
            label_2 = tuple(clip_labels.iloc[i + 1][["Visibility", "X", "Y"]])
            label_3 = tuple(clip_labels.iloc[i + 2][["Visibility", "X", "Y"]])
            samples.append((frame_path_1, frame_path_2, frame_path_3, label_1, label_2, label_3))
    return samples


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    log("Building samples...")
    samples = build_samples()
    log(f"Built {len(samples)} sliding-window triplets")

    dataset = TrackNetDataset(samples)
    heatmap_dataset = TrackNetHeatmapDataset(dataset)

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
        num_workers=0, pin_memory=True,
    )
    full_val_loader = torch.utils.data.DataLoader(
        val_subset_full, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = TrackNetV2().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        log("No checkpoint found, starting fresh.")

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
