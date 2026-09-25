import { useEffect, useState } from "react";
import InputPanel from "./components/InputPanel.jsx";
import PipelinePanel from "./components/PipelinePanel.jsx";
import OutputPanel from "./components/OutputPanel.jsx";
import { STAGES, runPipeline } from "./api.js";

const initialStages = () =>
  Object.fromEntries(STAGES.map((s) => [s.id, { status: "queued", progress: 0 }]));

export default function App() {
  const [video, setVideo] = useState(null); // { file, name, size, url }
  const [runState, setRunState] = useState("idle"); // idle | running | done | error
  const [stages, setStages] = useState(initialStages);
  const [log, setLog] = useState([]);
  const [result, setResult] = useState(null);

  useEffect(() => () => video?.url && URL.revokeObjectURL(video.url), [video]);

  const selectVideo = (next) => {
    setVideo(next);
    setRunState("idle");
    setStages(initialStages());
    setLog([]);
    setResult(null);
  };

  const run = async () => {
    if (!video || runState === "running") return;
    setRunState("running");
    setStages(initialStages());
    setLog([]);
    setResult(null);
    const stamp = () => new Date().toLocaleTimeString([], { hour12: false });
    try {
      const res = await runPipeline(video.file ?? video, {
        onStage: (id, patch) =>
          setStages((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } })),
        onLog: (line) => setLog((prev) => [...prev, `[${stamp()}] ${line}`]),
      });
      setResult(res);
      setRunState("done");
    } catch (err) {
      setLog((prev) => [...prev, `[${stamp()}] Error: ${err.message}`]);
      setStages((prev) =>
        Object.fromEntries(
          Object.entries(prev).map(([id, s]) => [
            id,
            s.status === "running" ? { ...s, status: "error" } : s,
          ]),
        ),
      );
      setRunState("error");
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1 className="brand">CREASE</h1>
        <span className="tagline">Single-camera ball tracking</span>
      </header>
      <main className="columns">
        <InputPanel video={video} runState={runState} onSelect={selectVideo} />
        <PipelinePanel
          canRun={!!video && runState !== "running"}
          running={runState === "running"}
          stages={stages}
          log={log}
          onRun={run}
        />
        <OutputPanel result={result} runState={runState} />
      </main>
    </div>
  );
}
