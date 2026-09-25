// Pipeline client. Detection and the 3D reconstruction run on the Python server
// (server/app.py in the repo root), reached through the Vite proxy at /api.
// Spin estimation doesn't exist yet, so that stage stays queued.

export const STAGES = [
  { id: "detect", title: "Ball detection & tracking" },
  { id: "physics", title: "3D reconstruction (physics)" },
  { id: "spin", title: "Spin estimation" },
];

/**
 * @param {File} file
 * @param {{ onStage: (id, patch) => void, onLog: (line: string) => void }} cb
 * @returns {Promise<{ videoUrl: string, detections: object, flight: object|null }>}
 *   videoUrl is an H.264 copy of the upload served by the server (browsers
 *   can't play the mp4v clips OpenCV writes). detections is the server's
 *   JSON: { fps, width, height, n_frames, detections: [{ frame, t, u, v, conf }] }
 *   with u, v in video pixels. flight is the 3D result (server/reconstruct.py),
 *   or null when it couldn't be made.
 */
export async function runPipeline(file, { onStage, onLog }) {
  onLog(`Loaded ${file.name}`);
  onStage("detect", { status: "running", progress: 0.5 });
  onStage("physics", { status: "running", progress: 0.2 });
  onLog("Uploading to server: detection, then 3D reconstruction…");

  const body = new FormData();
  body.append("video", file);
  const res = await fetch("/api/detect", { method: "POST", body });
  const data = await res.json().catch(() => null);
  if (!res.ok || !data) {
    throw new Error(data?.error ?? `Server returned ${res.status}. Is server/app.py running?`);
  }

  onStage("detect", { status: "done", progress: 1 });
  onLog(
    `Ball detection & tracking: ball found in ${data.detections.length}/${data.n_frames} frames ` +
      `(${data.width}x${data.height} @ ${data.fps.toFixed(1)} fps)`,
  );

  const flight = data.reconstruction ?? null;
  if (flight) {
    onStage("physics", { status: "done", progress: 1 });
    onLog(
      `3D reconstruction: ${flight.n_flights} flight(s), ${flight.bounces.length} bounce(s), ` +
        `fit error ${flight.metrics.fit_error_px} px`,
    );
    if (flight.dropped_detections) {
      onLog(`3D reconstruction: left out ${flight.dropped_detections} detection(s) with no ball width`);
    }
    onLog("3D reconstruction: focal length and ball widths from the cricket-synth label");
  } else {
    onStage("physics", { status: "error", progress: 1 });
    onLog(data.reconstruction_error ?? "3D reconstruction: no result");
  }
  onLog("Spin estimation is not built yet.");
  return { videoUrl: data.video_url, detections: data, flight };
}
