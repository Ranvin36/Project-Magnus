"""
Small HTTP server for the frontend: upload a video, get the ball detections back.

Usage:
  python server/app.py
Then start the frontend (cd frontend && npm run dev). Vite forwards /api/* here.
"""
import os
import subprocess
import tempfile
import uuid

import imageio_ffmpeg
import torch
from flask import Flask, jsonify, request, send_from_directory

import detection
import pipeline

# Browser-playable copies of the uploads are kept here.
WEB_VIDEO_DIR = os.path.join(pipeline.ROOT, "output", "web")
os.makedirs(WEB_VIDEO_DIR, exist_ok=True)

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = detection.load_model(pipeline.DEFAULT_WEIGHTS, device)
print(f"Model loaded on {device}")


def to_browser_mp4(src, dst):
    """Re-encode to H.264. Browsers can't play OpenCV's mp4v (MPEG-4 Part 2) videos."""
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst,
    ], check=True)


@app.post("/api/detect")
def detect():
    video = request.files.get("video")
    if video is None:
        return jsonify(error="No video uploaded"), 400

    web_name = uuid.uuid4().hex + ".mp4"

    # OpenCV and ffmpeg need a file on disk, so save the upload to a temp folder first.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + os.path.splitext(video.filename)[1])
        video.save(path)
        try:
            result = detection.detect(path, model, device)
            to_browser_mp4(path, os.path.join(WEB_VIDEO_DIR, web_name))
        except (IOError, subprocess.CalledProcessError) as err:
            return jsonify(error=str(err)), 400

    data = pipeline.detections_to_dict(result)
    data["video"] = video.filename
    data["video_url"] = f"/api/videos/{web_name}"
    return jsonify(data)


@app.get("/api/videos/<name>")
def video_file(name):
    return send_from_directory(WEB_VIDEO_DIR, name)


if __name__ == "__main__":
    app.run(port=5000)
