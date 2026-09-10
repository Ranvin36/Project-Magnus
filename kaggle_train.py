"""
Kaggle-ready training script for TrackNetV2 ball detection.

Same model/dataset/checkpoint logic as resume_training.py, adapted to run
on Kaggle's free GPU notebooks:
  - Locates the raw TrackNetV2 video/csv dataset under /kaggle/input
    (wherever you attached it, whatever it got nested under).
  - Extracts frames into /kaggle/temp (ephemeral scratch space, NOT saved
    on commit -- keeps the committed output small and fast).
  - Checkpoints (training_checkpoint.pt / best_model.pt / training_log.txt)
    go to /kaggle/working, which IS saved when you commit a version.
  - On startup, if no checkpoint exists yet in /kaggle/working, it looks
    for one under any attached /kaggle/input/<slug>/ dataset (e.g. the
    previous version's own output, added back in as input) and copies it
    in before resuming. This is how you carry progress across Kaggle
    session time limits.

Setup (one time):
  1. Zip Dataset/TrackNetV2 (you already have TrackNetV2.zip at the repo
     root) and upload it as a new Kaggle Dataset.
  2. New Notebook -> Add Input -> your dataset. Settings -> Accelerator ->
     GPU (T4 x2 or P100). Internet can stay off.
  3. Paste this file into a single cell (or add it as a Notebook file) and
     run it, or click "Save Version -> Save & Run All (Commit)" to run in
     the background even after closing the tab.

To resume after a session's time limit is hit:
  1. Open the notebook version that just finished, note its output files
     (training_checkpoint.pt etc. under its Output tab).
  2. New session of the same notebook -> Add Input -> "Notebook Output
     Files" -> this notebook's latest version. Run again -- it auto-detects
     and resumes.
"""
import copy
import os
import random
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

WINDOW_SIZE = 3
HEATMAP_SIGMA = 5
VAL_FRACTION = 0.2
BATCH_SIZE = 8
POS_WEIGHT = 200.0
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 2
CHECKPOINT_EVERY = 200

ON_KAGGLE = os.path.isdir("/kaggle/input")
WORK_DIR = "/kaggle/working" if ON_KAGGLE else "Dataset"
SCRATCH_DIR = "/kaggle/temp" if ON_KAGGLE else WORK_DIR

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

FRAMES_BASE_OUTPUT = os.path.join(SCRATCH_DIR, "TrackNetV2_extracted_frames")
CHECKPOINT_PATH = os.path.join(WORK_DIR, "training_checkpoint.pt")
LOG_PATH = os.path.join(WORK_DIR, "training_log.txt")
BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def find_videos_dir(search_root):
    """Locate the TrackNetV2 root dir regardless of how it got nested
    under /kaggle/input -- looks for a .../<category>/<match>/csv/*_ball.csv
    file and walks back up 3 levels."""
    for dirpath, _dirnames, filenames in os.walk(search_root):
        if os.path.basename(dirpath) == "csv" and any(f.endswith("_ball.csv") for f in filenames):
            match_dir = os.path.dirname(dirpath)
            category_dir = os.path.dirname(match_dir)
            return os.path.dirname(category_dir)
    raise FileNotFoundError(
        f"Could not find a TrackNetV2 csv/ folder anywhere under {search_root}. "
        "Did you attach the dataset as a notebook input?"
    )


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing_frames = [f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if existing_frames:
        return len(existing_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: could not open video {video_path}")
        return 0

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    return i


def extract_all_frames(videos_dir, frames_base_output):
    log(f"Extracting frames: {videos_dir} -> {frames_base_output}")
    n_videos = 0
    for dirpath, _dirnames, filenames in os.walk(videos_dir):
        for filename in filenames:
            if not filename.endswith(".mp4"):
                continue
            video_file_path = os.path.join(dirpath, filename)
            relative_path = os.path.relpath(video_file_path, videos_dir)
            video_output_dir = os.path.join(
                frames_base_output, os.path.dirname(relative_path), os.path.splitext(filename)[0]
            )
            extract_frames(video_file_path, video_output_dir)
            n_videos += 1
            if n_videos % 20 == 0:
                log(f"  extracted {n_videos} clips so far...")
    log(f"Frame extraction done: {n_videos} clips.")


def restore_previous_checkpoint():
    """Cross-session resume: if /kaggle/working has no checkpoint yet,
    look for one in any attached input dataset (e.g. a previous version's
    own output re-attached as input) and copy it in."""
    if os.path.exists(CHECKPOINT_PATH):
        return
    if not os.path.isdir("/kaggle/input"):
        return
    for name in os.listdir("/kaggle/input"):
        candidate_dir = os.path.join("/kaggle/input", name)
        for dirpath, _dirnames, filenames in os.walk(candidate_dir):
            if "training_checkpoint.pt" in filenames:
                src = os.path.join(dirpath, "training_checkpoint.pt")
                log(f"Found previous checkpoint at {src}, copying into {WORK_DIR}...")
                shutil.copy(src, CHECKPOINT_PATH)
                for extra in ("best_model.pt", "training_log.txt"):
                    extra_src = os.path.join(dirpath, extra)
                    if os.path.exists(extra_src):
                        shutil.copy(extra_src, os.path.join(WORK_DIR, extra))
                return


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


def generate_heatmap(x, y, visibility, H, W, sigma=HEATMAP_SIGMA):
    if not visibility:
        return np.zeros((H, W), dtype=np.float32)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    heatmap = np.exp(-((cols - x) ** 2 + (rows - y) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


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


def build_samples(videos_dir, frames_base_output):
    samples = []
    for dirpath, dirnames, filenames in os.walk(frames_base_output):
        if dirnames:
            continue
        frame_files = sorted(f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not frame_files:
            continue

        rel_path = os.path.relpath(dirpath, frames_base_output)
        parts = rel_path.split(os.sep)
        clip_name = parts[-1]
        csv_parts = ["csv" if p == "video" else p for p in parts[:-1]]
        csv_path = os.path.join(videos_dir, *csv_parts, f"{clip_name}_ball.csv")

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
    log(f"Running on Kaggle: {ON_KAGGLE}")
    log(f"Work dir: {WORK_DIR}  Scratch dir: {SCRATCH_DIR}")

    if ON_KAGGLE:
        videos_dir = find_videos_dir("/kaggle/input")
    else:
        videos_dir = "Dataset/TrackNetV2"
    log(f"Videos dir: {videos_dir}")

    restore_previous_checkpoint()

    extract_all_frames(videos_dir, FRAMES_BASE_OUTPUT)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    log("Building samples...")
    samples = build_samples(videos_dir, FRAMES_BASE_OUTPUT)
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
        num_workers=2, pin_memory=True,
    )
    full_val_loader = torch.utils.data.DataLoader(
        val_subset_full, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
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
