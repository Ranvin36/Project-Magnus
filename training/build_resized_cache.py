"""
Build a pre-resized copy of the cricket-synth dataset at the network's input
size (512x288), so training doesn't decode full-resolution JPEGs (up to
2560x1440) three times per sample.

Output has the same layout as the source, so the notebook can point at either:
  <out>/clips/<clip_id>/f_%04d.jpg   -- frames resized with INTER_AREA (the
                                        same resize the notebook applies),
                                        stored as JPEG quality 95
  <out>/labels/<clip_id>.json        -- copied unchanged; centre_px stays in
                                        native pixels, the notebook converts
                                        it using camera.intrinsics

~2 GB instead of ~21 GB. Resumable: each finished clip gets a .done marker,
and re-running skips those clips. CACHE_COMPLETE.json is written only once
every clip is done -- the notebook won't use the cache before then.

Usage:
  python training/build_resized_cache.py
  python training/build_resized_cache.py --src <cricket-synth/out> --dst <cache dir> --workers 8
"""
import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool

import cv2

INPUT_SIZE = (512, 288)  # (width, height) -- must match the notebook's INPUT_SIZE
JPEG_QUALITY = 95
DONE_MARKER = ".done"
COMPLETE_MARKER = "CACHE_COMPLETE.json"  # notebook checks for this before using the cache


def build_clip(job):
    """Resize every frame of one clip. Returns (clip_id, n_frames, n_unreadable)."""
    src_root, dst_root, label_file = job
    clip_id = label_file[:-5]
    src_clip = os.path.join(src_root, "clips", clip_id)
    dst_clip = os.path.join(dst_root, "clips", clip_id)
    if os.path.exists(os.path.join(dst_clip, DONE_MARKER)):
        return clip_id, 0, 0
    if not os.path.isdir(src_clip):
        return clip_id, 0, 0

    os.makedirs(dst_clip, exist_ok=True)
    n_frames = n_unreadable = 0
    for name in sorted(os.listdir(src_clip)):
        if not name.endswith(".jpg"):
            continue
        image = cv2.imread(os.path.join(src_clip, name))
        if image is None:
            n_unreadable += 1
            continue
        resized = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(os.path.join(dst_clip, name), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            raise IOError(f"Failed to write {os.path.join(dst_clip, name)} -- disk full?")
        n_frames += 1

    shutil.copy2(os.path.join(src_root, "labels", label_file), os.path.join(dst_root, "labels", label_file))
    with open(os.path.join(dst_clip, DONE_MARKER), "w") as f:
        json.dump({"frames": n_frames, "unreadable": n_unreadable, "size": INPUT_SIZE,
                   "jpeg_quality": JPEG_QUALITY}, f)
    return clip_id, n_frames, n_unreadable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"D:\Git-Repos\MAGNUS\Ball-Tracking\cricket-synth\out")
    parser.add_argument("--dst", default=None, help="default: <src>_512x288 next to the source")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    dst = args.dst or os.path.normpath(args.src) + f"_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}"

    os.makedirs(os.path.join(dst, "clips"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels"), exist_ok=True)
    label_files = sorted(f for f in os.listdir(os.path.join(args.src, "labels")) if f.endswith(".json"))
    jobs = [(args.src, dst, f) for f in label_files]
    print(f"{len(jobs)} clips: {args.src} -> {dst} ({args.workers} workers)", flush=True)

    start = time.time()
    total_frames = total_unreadable = 0
    with Pool(args.workers) as pool:
        for i, (clip_id, n_frames, n_unreadable) in enumerate(pool.imap_unordered(build_clip, jobs), 1):
            total_frames += n_frames
            total_unreadable += n_unreadable
            if n_unreadable:
                print(f"  {clip_id}: {n_unreadable} unreadable frame(s) skipped", flush=True)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clips, {total_frames} frames written "
                      f"({time.time() - start:.0f}s)", flush=True)

    print(f"Done: {total_frames} frames written this run, {total_unreadable} unreadable, "
          f"{(time.time() - start) / 60:.1f} min")

    # The notebook only switches to the cache once this marker exists, so a
    # half-built cache is never silently trained on as if it were the full set.
    n_done = sum(os.path.exists(os.path.join(dst, "clips", f[:-5], DONE_MARKER)) for f in label_files)
    if n_done == len(label_files):
        with open(os.path.join(dst, COMPLETE_MARKER), "w") as f:
            json.dump({"clips": n_done, "size": INPUT_SIZE, "jpeg_quality": JPEG_QUALITY, "src": args.src}, f)
        print(f"Cache complete ({n_done} clips) -> wrote {COMPLETE_MARKER}")
    else:
        print(f"Cache INCOMPLETE: {n_done}/{len(label_files)} clips done -- re-run to finish.")


if __name__ == "__main__":
    main()
