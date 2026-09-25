import { useRef } from "react";
import TrackedVideo from "./TrackedVideo.jsx";
import FlightViews from "./FlightViews.jsx";

// What the 3D reconstruction reports. Spin isn't built yet, so it stays empty.
const METRICS = [
  { label: "Speed (first seen)", unit: "km/h", get: (f) => f.metrics.speed_first_seen_kmh },
  { label: "True speed (label)", unit: "km/h", get: (f) => f.true_speed_kmh?.toFixed(1) },
  { label: "Bounce distance", unit: "m", get: (f) => f.bounces[0]?.forward.toFixed(2) },
  { label: "Fit error", unit: "px", get: (f) => f.metrics.fit_error_px },
  { label: "Flights", get: (f) => f.n_flights },
  { label: "Spin rate", unit: "rpm", get: () => null },
];

function placeholderText(result, runState) {
  if (runState === "running") return "Processing…";
  if (runState === "error") return "Run failed. See the log.";
  return "Tracked video will appear here.";
}

export default function OutputPanel({ result, runState }) {
  const videoRef = useRef(null);
  const flight = result?.flight;

  return (
    <section className="panel panel--output">
      <h2 className="section-label">Output</h2>

      <div className="viewer">
        {result?.videoUrl && result.detections ? (
          <TrackedVideo src={result.videoUrl} data={result.detections} flight={flight}
            videoRef={videoRef} />
        ) : (
          <p className="viewer__empty">{placeholderText(result, runState)}</p>
        )}
      </div>

      {flight ? (
        <FlightViews flight={flight} fps={result.detections.fps} videoRef={videoRef} />
      ) : result?.detections?.reconstruction_error ? (
        <p className="flight__missing">{result.detections.reconstruction_error}</p>
      ) : null}

      <dl className="metrics">
        {METRICS.map(({ label, unit, get }) => {
          const value = flight ? get(flight) : null;
          return (
            <div className="metric" key={label}>
              <dt>{label}</dt>
              <dd>
                {value != null ? <span className="metric__value">{value}</span>
                  : <span className="metric__dash" aria-label="no value" />}
                {unit && <span className="metric__unit">{unit}</span>}
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
