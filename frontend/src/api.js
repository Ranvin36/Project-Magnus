// Pipeline client. Ball detection runs on the Python server (server/app.py in the
// repo root), reached through the Vite proxy at /api. Physics modelling and
// spin estimation don't exist yet, so those stages stay queued.

export const STAGES = [
  { id: "detect", title: "Ball detection & tracking" },
  { id: "physics", title: "Physics modelling" },
  { id: "spin", title: "Spin estimation" },
];

/**
 * @param {File} file
 * @param {{ onStage: (id, patch) => void, onLog: (line: string) => void }} cb
 * @returns {Promise<{ videoUrl: string, detections: object, metrics: object|null }>}
 *   videoUrl is an H.264 copy of the upload served by the server (browsers
 *   can't play the mp4v clips OpenCV writes). detections is the server's
 *   JSON: { fps, width, height, n_frames, detections: [{ frame, t, u, v, conf }] }
 *   with u, v in video pixels.
 */
export async function runPipeline(file, { onStage, onLog }) {
  onLog(`Loaded ${file.name}`);
  onStage("detect", { status: "running", progress: 0.5 });
  onLog("Ball detection & tracking: uploading to server…");

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
  onLog("Physics modelling and spin estimation are not built yet.");
  return { videoUrl: data.video_url, detections: data, metrics: null };
}
