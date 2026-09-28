"""
Small HTTP server for the frontend: upload a video, get the ball detections back.
Also GET /api/simulate, which runs physics.backproject() on fixed test inputs.

Usage:
  python server/app.py
Then start the frontend (cd frontend && npm run dev). Vite forwards /api/* here.
"""
import importlib
import os
import subprocess
import tempfile
import traceback
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


# Test inputs for step 1, hardcoded from cricket-synth label main1000_000009.json.
# true_Z is the label's depth_m (the real distance from the camera), for comparing.
TEST_CAMERA = dict(fx=2743.353363255959, cx=960.0, cy=540.0)
TEST_FRAMES = [
    dict(frame=22, u=756.9, v=152.6, radius_px=13.07, true_Z=7.513),
    dict(frame=30, u=867.5, v=510.0, radius_px=6.56, true_Z=14.979),
    dict(frame=40, u=907.0, v=629.2, radius_px=4.22, true_Z=23.298),
]


@app.get("/api/simulate")
def simulate():
    """Call physics.backproject(u, v, radius_px, fx, cx, cy) on each test frame
    and return what it gives back. physics.py is re-imported on every request,
    so edits to it show up without restarting the server."""
    try:
        import physics
        importlib.reload(physics)
        results = []
        for f in TEST_FRAMES:
            output = physics.backproject(f["u"], f["v"], f["radius_px"], **TEST_CAMERA)
            results.append(dict(f, output=output))
        return jsonify(camera=TEST_CAMERA, results=results)
    except Exception:
        # Send the full error back, so it can be read from the response.
        return jsonify(error=traceback.format_exc()), 500


@app.get("/api/videos/<name>")
def video_file(name):
    return send_from_directory(WEB_VIDEO_DIR, name)


if __name__ == "__main__":
    app.run(port=5000)
