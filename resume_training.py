"""
Shared TrackNetV2 model/dataset definitions.

Originally the standalone shuttlecock (TrackNetV2 dataset) training script;
that dataset and training driver have been retired now that the project
targets the cricket-synth dataset only (see train_cricket_synth.py). What
remains here is the dataset-agnostic core -- model architecture, frame/label
loading, and Gaussian-heatmap generation -- imported as `resume_training as rt`
by train_cricket_synth.py and check_predictions.py.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn

WINDOW_SIZE = 3
HEATMAP_SIGMA = 5
VAL_FRACTION = 0.2
POS_WEIGHT = 200.0


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
