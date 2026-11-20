"use client";

import { useEffect, useState } from "react";
import { api, ExperimentRun } from "@/lib/api";

export default function ResultsPage() {
  const [runId, setRunId] = useState("");
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [results, setResults] = useState<Record<string, unknown>[]>([]);
  const [timings, setTimings] = useState<Record<string, number>>({});
  const [confusion, setConfusion] = useState<
    Record<string, Record<string, number[][]>>
  >({});
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listRuns()
      .then((list) => {
        setRuns(list.filter((r) => r.status === "succeeded"));
        const last = list.find((r) => r.status === "succeeded");
        if (last) setRunId(last.id);
      })
      .catch((e) => setError(String(e.message || e)));
  }, []);

  useEffect(() => {
    if (!runId) return;
    Promise.all([api.results(runId), api.timings(runId), api.confusion(runId)])
      .then(([r, t, c]) => {
        setResults(r.results);
        setTimings(t.timings);
        setConfusion(c.confusion);
      })
      .catch((e) => setError(String(e.message || e)));
  }, [runId]);

  return (
    <section className="panel stack">
      <div className="page-intro">
        <p className="section-label">Section · Results</p>
        <h1>Held-out evaluation metrics</h1>
        <p>
          Accuracy, weighted precision / recall / F1, training time, and confusion
          matrices for each classifier on SST-2, SST-2 validation, and IMDB.
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

      <h2>Table 1 · Classification metrics</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th>Accuracy</th>
              <th>F1</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row, i) => (
              <tr key={i}>
                <td>{String(row.Model)}</td>
                <td>{String(row.Dataset)}</td>
                <td>{Number(row.Accuracy).toFixed(3)}</td>
                <td>{Number(row["F1-score"]).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Table 2 · Training time (seconds)</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Seconds</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(timings).map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td>{v.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Figure · Confusion matrices</h2>
      <p className="muted">Rows: true class; columns: predicted class (0 = negative, 1 = positive).</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Dataset</th>
              <th>Model</th>
              <th>Matrix</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(confusion).flatMap(([ds, models]) =>
              Object.entries(models).map(([name, cm]) => (
                <tr key={`${ds}-${name}`}>
                  <td>{ds}</td>
                  <td>{name}</td>
                  <td>
                    <div className="cm">
                      {cm.flat().map((n, i) => (
                        <span key={i}>{n}</span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
