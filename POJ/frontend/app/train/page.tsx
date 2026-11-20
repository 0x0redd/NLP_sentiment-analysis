"use client";

import { useEffect, useState } from "react";
import { api, ExperimentRun } from "@/lib/api";

export default function TrainPage() {
  const [fastMode, setFastMode] = useState(true);
  const [runDistilbert, setRunDistilbert] = useState(false);
  const [runAblation, setRunAblation] = useState(true);
  const [runNegation, setRunNegation] = useState(true);
  const [run, setRun] = useState<ExperimentRun | null>(null);
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refreshRuns() {
    const list = await api.listRuns();
    setRuns(list);
  }

  useEffect(() => {
    refreshRuns().catch((e) => setError(String(e.message || e)));
  }, []);

  useEffect(() => {
    if (!run || (run.status !== "queued" && run.status !== "running")) return;
    const t = setInterval(async () => {
      try {
        const latest = await api.getRun(run.id);
        setRun(latest);
        if (latest.status === "succeeded" || latest.status === "failed") {
          refreshRuns();
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 2000);
    return () => clearInterval(t);
  }, [run?.id, run?.status]);

  async function start() {
    setBusy(true);
    setError("");
    try {
      const created = await api.createRun({
        fast_mode: fastMode,
        run_distilbert: runDistilbert,
        run_ablation: runAblation,
        run_negation: runNegation,
      });
      setRun(created);
      await refreshRuns();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function activate(id: string) {
    setError("");
    try {
      await api.activate(id);
      await refreshRuns();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <section className="panel">
      <div className="page-intro">
        <p className="section-label">Section · Experiments</p>
        <h1>Retrain and activate model artefacts</h1>
        <p>
          Launch a full or partial reproduction of the merged notebook pipeline.
          Successful runs write vectorisers, classifiers, metrics, and optional
          DistilBERT weights. Activate a run to load it into the inference registry.
        </p>
      </div>
      <div className="grid-2">
        <div className="stack">
          <div className="checks">
            <label>
              <input type="checkbox" checked={fastMode} onChange={(e) => setFastMode(e.target.checked)} />
              FAST_MODE — 8,000 train / 2,000 test rows
            </label>
            <label>
              <input
                type="checkbox"
                checked={runDistilbert}
                onChange={(e) => setRunDistilbert(e.target.checked)}
              />
              Fine-tune DistilBERT (prefer GPU)
            </label>
            <label>
              <input
                type="checkbox"
                checked={runAblation}
                onChange={(e) => setRunAblation(e.target.checked)}
              />
              Feature-representation ablation
            </label>
            <label>
              <input
                type="checkbox"
                checked={runNegation}
                onChange={(e) => setRunNegation(e.target.checked)}
              />
              Negation-aware logistic regression
            </label>
          </div>
          <button className="btn primary" disabled={busy} onClick={start}>
            {busy ? "Submitting…" : "Start experiment"}
          </button>
          {run && (
            <div className="stack">
              <p>
                Run ID <code>{run.id}</code>{" "}
                <span className="pill">{run.status}</span>
              </p>
              <p className="muted">{run.progress || "Awaiting progress…"}</p>
              {run.error && <pre className="error">{run.error}</pre>}
            </div>
          )}
          {error && <p className="error">{error}</p>}
        </div>
        <div className="stack">
          <h2>Experiment log</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {runs.length === 0 && (
                  <tr>
                    <td colSpan={3}>No runs recorded.</td>
                  </tr>
                )}
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td>{r.status}</td>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                    <td>
                      {r.status === "succeeded" && (
                        <button className="btn" onClick={() => activate(r.id)}>
                          Activate
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
