import TrackedVideo from "./TrackedVideo.jsx";

const METRICS = [
  { key: "releaseSpeed", label: "Release speed", unit: "km/h" },
  { key: "pitchSpeed", label: "Speed at pitch", unit: "km/h" },
  { key: "bouncePoint", label: "Bounce point", unit: "m" },
  { key: "length", label: "Length" },
  { key: "spinRate", label: "Spin rate", unit: "rpm" },
  { key: "deviation", label: "Deviation" },
];

function placeholderText(result, runState) {
  if (runState === "running") return "Processing…";
  if (runState === "error") return "Run failed. See the log.";
  return "Tracked video will appear here.";
}

export default function OutputPanel({ result, runState }) {
  const metrics = result?.metrics ?? {};

  return (
    <section className="panel panel--output">
      <h2 className="section-label">Output</h2>

      <div className="viewer">
        {result?.videoUrl && result.detections ? (
          <TrackedVideo src={result.videoUrl} data={result.detections} />
        ) : (
          <p className="viewer__empty">{placeholderText(result, runState)}</p>
        )}
      </div>

      <dl className="metrics">
        {METRICS.map(({ key, label, unit }) => {
          const value = metrics[key];
          return (
            <div className="metric" key={key}>
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
