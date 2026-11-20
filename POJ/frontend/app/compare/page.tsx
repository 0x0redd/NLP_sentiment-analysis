"use client";

import { useEffect, useState } from "react";
import { api, ExperimentRun } from "@/lib/api";

export default function ComparePage() {
  const [runId, setRunId] = useState("");
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [mcnemar, setMcnemar] = useState<Record<string, unknown>[]>([]);
  const [controlled, setControlled] = useState<Record<string, unknown>[]>([]);
  const [ablation, setAblation] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listRuns()
      .then((list) => {
        const ok = list.filter((r) => r.status === "succeeded");
        setRuns(ok);
        if (ok[0]) setRunId(ok[0].id);
      })
      .catch((e) => setError(String(e.message || e)));
  }, []);

  useEffect(() => {
    if (!runId) return;
    Promise.all([api.mcnemar(runId), api.controlled(runId), api.ablation(runId)])
      .then(([m, c, a]) => {
        setMcnemar(m.mcnemar);
        setControlled(c.controlled);
        setAblation(a.ablation);
      })
      .catch((e) => setError(String(e.message || e)));
  }, [runId]);

  const controlledCols =
    controlled[0] != null
      ? Object.keys(controlled[0]).filter((k) => k !== "Text")
      : [];

  return (
    <section className="panel stack">
      <div className="page-intro">
        <p className="section-label">Section · Analysis</p>
        <h1>Statistical comparison and qualitative probes</h1>
        <p>
          McNemar exact tests with Holm correction assess whether paired models
          disagree significantly on the same examples. Controlled phrases illustrate
          negation sensitivity. Feature ablations compare TF-IDF, bigrams, bag-of-words,
          and character n-grams under identical classifiers.
        </p>
      </div>
      <div>
        <label htmlFor="run">Experiment run</label>
        <select id="run" value={runId} onChange={(e) => setRunId(e.target.value)}>
          {runs.length === 0 && <option value="">No succeeded runs</option>}
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id.slice(0, 8)}… · {new Date(r.created_at).toLocaleString()}
            </option>
          ))}
        </select>
      </div>
      {error && <p className="error">{error}</p>}

      <h2>Table · McNemar with Holm adjustment</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model 1</th>
              <th>Model 2</th>
              <th>Dataset</th>
              <th>p Holm</th>
              <th>Sig. (5%)</th>
            </tr>
          </thead>
          <tbody>
            {mcnemar.length === 0 && (
              <tr>
                <td colSpan={5}>No McNemar rows for this run.</td>
              </tr>
            )}
            {mcnemar.map((row, i) => (
              <tr key={i}>
                <td>{String(row["Model 1"])}</td>
                <td>{String(row["Model 2"])}</td>
                <td>{String(row.Dataset)}</td>
                <td>{Number(row["p Holm"]).toFixed(4)}</td>
                <td>{String(row["Significant (Holm 5%)"])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Table · Controlled phrases — P(positive)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Text</th>
              {controlledCols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {controlled.length === 0 && (
              <tr>
                <td colSpan={1 + controlledCols.length}>No controlled-phrase data.</td>
              </tr>
            )}
            {controlled.map((row, i) => (
              <tr key={i}>
                <td>{String(row.Text)}</td>
                {controlledCols.map((c) => (
                  <td key={c}>{Number(row[c]).toFixed(3)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Table · Feature ablation (SST-2 / IMDB)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Representation</th>
              <th>Model</th>
              <th>Dataset</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {ablation.filter((r) => r.Dataset === "SST-2" || r.Dataset === "IMDB").length ===
              0 && (
              <tr>
                <td colSpan={4}>No ablation rows (disabled for this run).</td>
              </tr>
            )}
            {ablation
              .filter((r) => r.Dataset === "SST-2" || r.Dataset === "IMDB")
              .slice(0, 32)
              .map((row, i) => (
                <tr key={i}>
                  <td>{String(row.Representation)}</td>
                  <td>{String(row.Model)}</td>
                  <td>{String(row.Dataset)}</td>
                  <td>{Number(row.Accuracy).toFixed(3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
