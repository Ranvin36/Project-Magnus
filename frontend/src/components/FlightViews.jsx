import { useEffect, useRef } from "react";

// Two flat views of the 3D flight from server/reconstruct.py:
//   side view: distance travelled (forward) vs height above the ground
//   top view:  distance travelled (forward) vs sideways drift (right)
// Each shows the step 1 points (distance from ball size alone), the physics
// fit through them, the true path when known, and a dot synced to the video.

const VIEWS = [
  { key: "height", label: "Side view", axis: "height (m)", floorAtZero: true },
  { key: "right", label: "Top view", axis: "sideways (m)" },
];
const PAD = { left: 40, right: 12, top: 12, bottom: 24 };

function niceStep(range) {
  const raw = range / 5;
  const pow = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw);
}

function drawView(canvas, flight, view, frame) {
  const css = getComputedStyle(canvas);
  const color = (name) => css.getPropertyValue(name).trim();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== w * dpr) canvas.width = w * dpr;
  if (canvas.height !== h * dpr) canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const series = [flight.frames, flight.measured, flight.truth ?? []];
  const all = series.flat();
  const xs = all.map((p) => p.forward);
  const ys = all.map((p) => p[view.key]);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (view.floorAtZero) y0 = Math.min(0, y0);
  const yPad = Math.max(0.2, (y1 - y0) * 0.1);
  [y0, y1] = [y0 - (view.floorAtZero ? 0 : yPad), y1 + yPad];

  const px = (x) => PAD.left + ((x - x0) / (x1 - x0)) * (w - PAD.left - PAD.right);
  const py = (y) => h - PAD.bottom - ((y - y0) / (y1 - y0)) * (h - PAD.top - PAD.bottom);

  // Grid and axis labels
  ctx.font = `10px ${color("--mono")}`;
  ctx.lineWidth = 1;
  ctx.fillStyle = color("--ink-muted");
  const xStep = niceStep(x1 - x0);
  ctx.textAlign = "center";
  for (let x = Math.ceil(x0 / xStep) * xStep; x <= x1; x += xStep) {
    ctx.strokeStyle = color("--track");
    ctx.beginPath(); ctx.moveTo(px(x), PAD.top); ctx.lineTo(px(x), h - PAD.bottom); ctx.stroke();
    ctx.fillText(`${+x.toFixed(1)}`, px(x), h - PAD.bottom + 14);
  }
  const yStep = niceStep(y1 - y0);
  ctx.textAlign = "right";
  for (let y = Math.ceil(y0 / yStep) * yStep; y <= y1; y += yStep) {
    ctx.strokeStyle = Math.abs(y) < 1e-9 ? color("--rule") : color("--track");
    ctx.beginPath(); ctx.moveTo(PAD.left, py(y)); ctx.lineTo(w - PAD.right, py(y)); ctx.stroke();
    ctx.fillText(`${+y.toFixed(1)}`, PAD.left - 6, py(y) + 3);
  }

  const line = (points, dashed) => {
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, px(p.forward), py(p[view.key])));
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // True path
  if (flight.truth) {
    ctx.strokeStyle = color("--ink-soft");
    ctx.lineWidth = 1.5;
    line(flight.truth, true);
  }
  // Step 1 points: distance from ball size alone
  ctx.strokeStyle = color("--ink-muted");
  ctx.lineWidth = 1.2;
  for (const p of flight.measured) {
    ctx.beginPath();
    ctx.arc(px(p.forward), py(p[view.key]), 3.5, 0, Math.PI * 2);
    ctx.stroke();
  }
  // Physics fit
  ctx.strokeStyle = color("--accent");
  ctx.lineWidth = 2.5;
  line(flight.frames);

  // Bounces
  for (const b of flight.bounces) {
    const y = view.key === "height" ? 0 : b.right;
    ctx.fillStyle = color("--ink");
    ctx.beginPath();
    ctx.moveTo(px(b.forward), py(y) - 7);
    ctx.lineTo(px(b.forward) - 5, py(y) - 15);
    ctx.lineTo(px(b.forward) + 5, py(y) - 15);
    ctx.fill();
    ctx.textAlign = "left";
    ctx.fillText(`bounce ${b.forward.toFixed(1)} m`, px(b.forward) + 8, py(y) - 10);
  }

  // Where the ball is at the current video frame
  const now = flight.frames.find((p) => p.frame === frame);
  if (now) {
    ctx.fillStyle = color("--accent");
    ctx.beginPath();
    ctx.arc(px(now.forward), py(now[view.key]), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color("--bg");
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

export default function FlightViews({ flight, fps, videoRef }) {
  const canvases = useRef([]);
  const readout = useRef(null);

  useEffect(() => {
    let raf;
    let lastFrame = null;
    let lastSize = "";
    const tick = () => {
      const video = videoRef.current;
      const frame = video ? Math.floor(video.currentTime * fps) : null;
      const size = canvases.current.map((c) => c && `${c.clientWidth}x${c.clientHeight}`).join();
      if (frame !== lastFrame || size !== lastSize) {
        lastFrame = frame;
        lastSize = size;
        VIEWS.forEach((view, i) => canvases.current[i] && drawView(canvases.current[i], flight, view, frame));
        const now = flight.frames.find((p) => p.frame === frame);
        if (readout.current) {
          readout.current.textContent = now
            ? `frame ${frame}: ${now.forward.toFixed(2)} m forward, ` +
              `${now.right.toFixed(2)} m sideways, ${now.height.toFixed(2)} m high`
            : `frame ${frame ?? "–"}: ball not in flight`;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [flight, fps, videoRef]);

  return (
    <div className="flight">
      {VIEWS.map((view, i) => (
        <figure className="flight__view" key={view.key}>
          <figcaption className="flight__caption">
            <span>{view.label}</span>
            <span className="flight__axis">{view.axis} vs distance (m)</span>
          </figcaption>
          <canvas ref={(el) => (canvases.current[i] = el)} className="flight__canvas" />
        </figure>
      ))}
      <p className="flight__readout" ref={readout} />
      <ul className="flight__legend">
        <li><span className="swatch swatch--fit" />Physics fit</li>
        <li><span className="swatch swatch--measured" />Step 1: distance from ball size</li>
        {flight.truth && <li><span className="swatch swatch--truth" />True path (label)</li>}
      </ul>
    </div>
  );
}
