// Pipeline client. There is no backend yet, so runPipeline simulates the
// three stages and returns no tracked video or metrics. Swap the body of
// runPipeline for a real call (e.g. POST the file to a FastAPI server that
// wraps the notebook's tracker) once one exists -- the UI only depends on
// the callbacks and the returned shape below.

export const STAGES = [
  { id: "detect", title: "Ball detection & tracking" },
  { id: "physics", title: "Physics modelling" },
  { id: "spin", title: "Spin estimation" },
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * @param {File|{name:string,size:number}} file
 * @param {{ onStage: (id, patch) => void, onLog: (line: string) => void }} cb
 * @returns {Promise<{ videoUrl: string|null, metrics: object|null, mock: boolean }>}
 */
export async function runPipeline(file, { onStage, onLog }) {
  onLog(`Loaded ${file.name}`);
  onLog("Backend not connected -- running simulated pipeline.");

  for (const stage of STAGES) {
    onStage(stage.id, { status: "running", progress: 0 });
    onLog(`${stage.title}: started`);
    for (let p = 1; p <= 10; p++) {
      await sleep(120);
      onStage(stage.id, { progress: p / 10 });
    }
    onStage(stage.id, { status: "done", progress: 1 });
    onLog(`${stage.title}: done`);
  }

  onLog("Simulated run complete. No results -- connect the backend.");
  return { videoUrl: null, metrics: null, mock: true };
}
