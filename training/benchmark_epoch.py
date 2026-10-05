"""
Benchmark how long one training epoch takes on the current machine, for the
model and data pipeline in notebooks/ball_detection_tracking.ipynb.

Measures the two things an epoch is made of, separately:
  1. GPU: train-step (fwd + bwd + clip + Adam, AMP) and eval-forward time per
     sample, on random 9x288x512 inputs, for each batch size that fits.
  2. CPU: data-loading time per sample (3 JPEG decodes + resize + heatmaps),
     for the notebook's current pipeline (full-resolution JPEGs) and for a
     simulated pre-resized 512x288 cache.
and combines them into per-epoch estimates.

The model, dataset and heatmap code are exec'd straight out of the notebook,
so the benchmark always matches what the notebook trains.

Usage (local or on a rented GPU box, from the repo root):
  python training/benchmark_epoch.py --data-root <path to cricket-synth/out>
  python training/benchmark_epoch.py --data-root ... --json results.json
  # Kaggle (script + notebook uploaded as inputs, dataset attached):
  !python benchmark_epoch.py --data-root /kaggle/input/<dataset>/out \n      --notebook ball_detection_tracking.ipynb --batch-sizes 4 8 16 32

On Windows, a batch that doesn't fit in VRAM spills into system RAM instead of
raising out-of-memory, and that step crawls. On a 4GB card pass
--batch-sizes 4 8.
"""
import argparse
import json
import os
import platform
import random
import time

import cv2
import numpy as np
import torch

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "notebooks", "ball_detection_tracking.ipynb")
# Sizes of the 70/15/15 split (see the split cell's output in the notebook).
TRAIN_SAMPLES = 49594
VAL_SAMPLES = 11007


def load_notebook_code(data_root, notebook=NOTEBOOK):
    """Exec the notebook cells that define the data pipeline and the model."""
    with open(notebook, encoding="utf-8") as f:
        cells = ["".join(c["source"]) for c in json.load(f)["cells"] if c["cell_type"] == "code"]

    def cell_with(marker):
        matches = [c for c in cells if marker in c]
        assert len(matches) == 1, f"expected one code cell containing {marker!r}, found {len(matches)}"
        return matches[0]

    g = {"__name__": "notebook"}
    exec(cell_with("import torch.nn as nn"), g)
    g["CLIPS_DIR"] = os.path.join(data_root, "clips")
    g["LABELS_DIR"] = os.path.join(data_root, "labels")
    for marker in ("def build_samples_cricket_synth", "class TrackNetDataset", "def generate_heatmap",
                   "class TrackNetHeatmapDataset", "class TrackNetV2"):
        exec(cell_with(marker), g)
    return g


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_gpu(g, device, batch_sizes, steps, warmup):
    model = g["TrackNetV2"]().to(device)
    heatmap_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(200.0, device=device))
    vis_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    results = []

    for bs in batch_sizes:
        try:
            x = torch.rand(bs, 9, 288, 512, device=device)
            hm = torch.zeros(bs, 3, 288, 512, device=device)
            hm[::2, :, 100, 200] = 1.0
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            model.train()
            for i in range(warmup + steps):
                if i == warmup:
                    sync(device)
                    t0 = time.perf_counter()
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    heat_logits, vis_logits = model(x)
                loss = heatmap_criterion(heat_logits.float(), hm) + 0.1 * vis_criterion(
                    vis_logits.float(), g["visibility_targets"](hm))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            sync(device)
            train_per_sample = (time.perf_counter() - t0) / (steps * bs)

            model.eval()
            with torch.no_grad():
                for i in range(warmup + steps):
                    if i == warmup:
                        sync(device)
                        t0 = time.perf_counter()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        model(x)
                sync(device)
            eval_per_sample = (time.perf_counter() - t0) / (steps * bs)

            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
            results.append({"batch_size": bs, "train_s_per_sample": train_per_sample,
                            "eval_s_per_sample": eval_per_sample, "peak_vram_gb": peak_gb})
            print(f"  batch {bs:>3}: train {1 / train_per_sample:7.1f} samples/s   "
                  f"eval {1 / eval_per_sample:7.1f} samples/s   peak VRAM {peak_gb:.2f} GB", flush=True)
        except torch.OutOfMemoryError:
            print(f"  batch {bs:>3}: out of memory -- stopping here", flush=True)
            break
        finally:
            x = hm = None
    return results


def bench_loading(g, n_samples, seed=0):
    samples = g["samples"]  # built when the notebook's sample-building cell was exec'd
    picked = random.Random(seed).sample(samples, n_samples)
    heatmap_dataset = g["TrackNetHeatmapDataset"](g["TrackNetDataset"](picked))

    # Current pipeline: full-resolution JPEGs from disk. The first pass warms
    # the OS file cache for these files so both pipelines are compared on
    # decode/resize cost rather than on cold-disk reads.
    for i in range(min(10, n_samples)):
        heatmap_dataset[i]
    t0 = time.perf_counter()
    for i in range(n_samples):
        heatmap_dataset[i]
    full_res = (time.perf_counter() - t0) / n_samples

    # Pre-resized cache: the same frames resized to 512x288 once and stored as
    # JPEG (q95), then decoded + turned into heatmaps per sample.
    resize_to = heatmap_dataset.base_dataset.resize_to
    W, H = resize_to
    encoded = []
    for s in picked:
        frames = []
        for path in s[:3]:
            img = cv2.resize(cv2.imread(path), resize_to, interpolation=cv2.INTER_AREA)
            frames.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
        encoded.append((frames, s[3:]))
    t0 = time.perf_counter()
    for frames, labels in encoded:
        imgs = [cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                for buf in frames]
        torch.from_numpy(np.concatenate([im.transpose(2, 0, 1) for im in imgs], axis=0))
        # Gaussian placement doesn't affect cost, so native coords aren't rescaled here.
        torch.from_numpy(np.stack([g["generate_heatmap"](x % W, y % H, v, H, W) for v, x, y in labels]))
    cached = (time.perf_counter() - t0) / n_samples

    print(f"  full-res JPEGs (current notebook): {1 / full_res:6.1f} samples/s per CPU worker")
    print(f"  pre-resized 512x288 cache:         {1 / cached:6.1f} samples/s per CPU worker")
    return {"full_res_s_per_sample": full_res, "cached_s_per_sample": cached}


def epoch_minutes(load_s, gpu_train_s, gpu_eval_s, workers):
    """workers=0: loading and GPU run back to back (notebook today).
    workers>0: loading overlaps the GPU, and the slower of the two sets the pace.
    Assumes loading scales linearly with workers, which needs that many free cores."""
    if workers == 0:
        total = TRAIN_SAMPLES * (load_s + gpu_train_s) + VAL_SAMPLES * (load_s + gpu_eval_s)
    else:
        total = (TRAIN_SAMPLES * max(load_s / workers, gpu_train_s)
                 + VAL_SAMPLES * max(load_s / workers, gpu_eval_s))
    return total / 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\Git-Repos\MAGNUS\Ball-Tracking\cricket-synth\out")
    parser.add_argument("--notebook", default=NOTEBOOK,
                        help="path to ball_detection_tracking.ipynb (default: the repo copy)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--load-samples", type=int, default=150)
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader workers to assume for the overlapped estimate (default: CPU count - 1, max 8)")
    parser.add_argument("--json", help="also write raw results to this file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    cpu_count = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else max(1, min(8, cpu_count - 1))
    print(f"GPU: {gpu_name}   CPU threads: {cpu_count}   torch {torch.__version__}   {platform.platform()}")

    g = load_notebook_code(args.data_root, args.notebook)
    torch.backends.cudnn.benchmark = True

    # Loading runs first: once CUDA is initialised it holds a lot of host
    # memory, which can starve the full-res JPEG decodes on small machines.
    print(f"\nData loading ({args.load_samples} random real triplets, single process):")
    load = bench_loading(g, args.load_samples)
    print("\nGPU compute (AMP, synthetic 9x288x512 inputs):")
    gpu = bench_gpu(g, device, args.batch_sizes, args.steps, args.warmup)

    bs4 = next(r for r in gpu if r["batch_size"] == 4)
    best = min(gpu, key=lambda r: r["train_s_per_sample"])
    scenarios = [
        ("notebook today: full-res JPEGs, 0 workers, batch 4", load["full_res_s_per_sample"], bs4, 0),
        (f"full-res JPEGs, {workers} workers, batch 4", load["full_res_s_per_sample"], bs4, workers),
        (f"pre-resized cache, {workers} workers, batch 4", load["cached_s_per_sample"], bs4, workers),
        (f"pre-resized cache, {workers} workers, batch {best['batch_size']}", load["cached_s_per_sample"], best, workers),
    ]
    print(f"\nEstimated epoch time ({TRAIN_SAMPLES} train + {VAL_SAMPLES} val samples):")
    estimates = []
    for name, load_s, gpu_r, w in scenarios:
        minutes = epoch_minutes(load_s, gpu_r["train_s_per_sample"], gpu_r["eval_s_per_sample"], w)
        bound = "GPU-bound" if w and gpu_r["train_s_per_sample"] >= load_s / w else "CPU-bound"
        print(f"  {minutes:7.1f} min  ({bound if w else 'serial'})  {name}")
        estimates.append({"scenario": name, "epoch_minutes": minutes})
    print(f"\nGPU-only floor (infinitely fast loading, batch {best['batch_size']}): "
          f"{epoch_minutes(0.0, best['train_s_per_sample'], best['eval_s_per_sample'], 1):.1f} min/epoch")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"gpu": gpu_name, "cpu_threads": cpu_count, "workers_assumed": workers,
                       "gpu_results": gpu, "loading": load, "estimates": estimates}, f, indent=2)


if __name__ == "__main__":
    main()
