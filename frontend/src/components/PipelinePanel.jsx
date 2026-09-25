import { useEffect, useRef } from "react";
import { STAGES } from "../api.js";

const STATUS_LABEL = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  error: "Failed",
};

function ArrowIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 12h14" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  );
}

export default function PipelinePanel({ canRun, running, stages, log, onRun }) {
  const logRef = useRef(null);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log]);

  return (
    <section className="panel panel--pipeline">
      <div className="pipeline__head">
        <h2 className="section-label">Pipeline</h2>
        <button className="btn-run" onClick={onRun} disabled={!canRun}>
          <span>{running ? "Running…" : "Run tracking"}</span>
          <ArrowIcon />
        </button>
      </div>

      <ol className="stages">
        {STAGES.map((stage, i) => {
          const s = stages[stage.id];
          return (
            <li key={stage.id} className={`stage stage--${s.status}`}>
              <span className="stage__num">{String(i + 1).padStart(2, "0")}</span>
              <div className="stage__body">
                <div className="stage__top">
                  <span className="stage__title">{stage.title}</span>
                  <span className="stage__status">{STATUS_LABEL[s.status]}</span>
                </div>
                <div className="stage__bar" role="progressbar"
                  aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(s.progress * 100)}>
                  <div className="stage__fill" style={{ width: `${s.progress * 100}%` }} />
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      <div className="log">
        <h2 className="section-label">Log</h2>
        <pre className="log__body" ref={logRef}>
          {log.length ? log.join("\n") : "Waiting for a run."}
        </pre>
      </div>
    </section>
  );
}
