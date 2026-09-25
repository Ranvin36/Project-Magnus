import { useRef, useState } from "react";

const SAMPLE_URL = "/sample.mp4";
const SAMPLE_NAME = "main1000_000009_test.mp4";

function formatSize(bytes) {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

const STATUS_TEXT = {
  idle: "Ready",
  running: "Processing",
  done: "Processed",
  error: "Failed",
};

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 15V4" />
      <path d="M7.5 8.5 12 4l4.5 4.5" />
      <path d="M4 14v5h16v-5" />
    </svg>
  );
}

export default function InputPanel({ video, runState, onSelect }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const locked = runState === "running";

  const acceptFile = (file) => {
    if (!file || !file.type.startsWith("video/")) return;
    onSelect({ file, name: file.name, size: file.size, url: URL.createObjectURL(file) });
  };

  const useSample = async () => {
    setLoadingSample(true);
    try {
      const res = await fetch(SAMPLE_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      acceptFile(new File([blob], SAMPLE_NAME, { type: "video/mp4" }));
    } catch (err) {
      console.error("Could not load sample clip:", err);
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <section className="panel panel--input">
      <h2 className="section-label">Input</h2>

      <div
        className={`dropzone${dragOver ? " is-over" : ""}${video ? " has-video" : ""}`}
        role="button"
        tabIndex={0}
        aria-disabled={locked}
        onClick={() => !locked && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!locked && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!locked) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!locked) acceptFile(e.dataTransfer.files?.[0]);
        }}
      >
        {video ? (
          <video className="dropzone__preview" src={video.url} muted controls
            onClick={(e) => e.stopPropagation()} />
        ) : (
          <div className="dropzone__prompt">
            <UploadIcon />
            <p className="dropzone__title">Upload video</p>
            <p className="dropzone__hint">Drop a file or click to browse.</p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            acceptFile(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      <dl className="meta">
        <div className="meta__row">
          <dt>File</dt>
          <dd title={video?.name}>{video?.name ?? "—"}</dd>
        </div>
        <div className="meta__row">
          <dt>Size</dt>
          <dd>{video ? formatSize(video.size) : "—"}</dd>
        </div>
        <div className="meta__row">
          <dt>Status</dt>
          <dd>{video ? STATUS_TEXT[runState] : "No video"}</dd>
        </div>
      </dl>

      <button className="btn-outline" onClick={useSample} disabled={locked || loadingSample}>
        {loadingSample ? "Loading…" : "Use sample clip"}
      </button>
    </section>
  );
}
