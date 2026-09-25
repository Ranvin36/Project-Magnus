import { useEffect, useRef } from "react";

const TRAIL_FRAMES = 7;
const COLOR = "224, 85, 63"; // --accent

// Plays the uploaded clip and draws the server's ball detections over it on a
// canvas, so the browser never has to decode an OpenCV-encoded video.
export default function TrackedVideo({ src, data }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const byFrame = new Map(data.detections.map((d) => [d.frame, d]));
    let raf;

    const draw = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      // Match the canvas to its on-screen size, then work out where the
      // video sits inside it (object-fit: contain leaves letterbox bars).
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w) canvas.width = w;
      if (canvas.height !== h) canvas.height = h;
      const scale = Math.min(w / data.width, h / data.height);
      const offX = (w - data.width * scale) / 2;
      const offY = (h - data.height * scale) / 2;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      const frame = Math.floor(video.currentTime * data.fps);
      for (let f = frame - TRAIL_FRAMES; f <= frame; f++) {
        const d = byFrame.get(f);
        if (!d) continue;
        const x = offX + d.u * scale;
        const y = offY + d.v * scale;
        const age = (f - (frame - TRAIL_FRAMES)) / TRAIL_FRAMES; // 0 old .. 1 now

        ctx.beginPath();
        ctx.arc(x, y, 2 + 3 * age, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${COLOR}, ${0.2 + 0.8 * age})`;
        ctx.fill();

        if (f === frame) {
          ctx.beginPath();
          ctx.arc(x, y, 10, 0, Math.PI * 2);
          ctx.strokeStyle = `rgb(${COLOR})`;
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.font = "11px monospace";
          ctx.fillText(d.conf.toFixed(2), x + 13, y - 8);
        }
      }
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [data]);

  return (
    <div className="tracked">
      <video ref={videoRef} className="viewer__video" src={src} controls autoPlay muted loop />
      <canvas ref={canvasRef} className="tracked__overlay" />
    </div>
  );
}
