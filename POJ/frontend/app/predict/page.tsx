"use client";

import { useCallback, useEffect, useState } from "react";
import { api, Health, PredictResult, SampleText } from "@/lib/api";

type DatasetFilter = "any" | "SST-2" | "IMDB";

export default function PredictPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [text, setText] = useState("");
  const [preprocess, setPreprocess] = useState("article");
  const [datasetFilter, setDatasetFilter] = useState<DatasetFilter>("any");
  const [sampleMeta, setSampleMeta] = useState<SampleText | null>(null);
  const [manualMode, setManualMode] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [sampling, setSampling] = useState(false);

  const loadSample = useCallback(async (filter: DatasetFilter = datasetFilter) => {
    setSampling(true);
    setError("");
    setResult(null);
    try {
      const sample = await api.sample(filter);
      setText(sample.text);
      setSampleMeta(sample);
      setManualMode(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSampling(false);
    }
  }, [datasetFilter]);

  useEffect(() => {
    api.health().then(setHealth).catch((e) => setError(String(e.message || e)));
    // Initial example from the dataset pool
    void loadSample("any");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onPredict() {
    setLoading(true);
    setError("");
    try {
      const res = await api.predict({ text, preprocess });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function onManualEdit(value: string) {
    setText(value);
    setManualMode(true);
    setSampleMeta(null);
    setResult(null);
  }

  return (
    <section className="panel">
      <div className="page-intro">
        <p className="section-label">Section · Inference</p>
        <h1>Model predictions on input text</h1>
        <p>
          Draw a random example from SST-2 or IMDB, or type your own sentence.
          Each loaded classifier returns polarity and P(positive).
        </p>
      </div>
      <div className="grid-2">
        <div className="stack">
          <div>
            <label htmlFor="dataset">Example source</label>
            <select
              id="dataset"
              value={datasetFilter}
              onChange={(e) => setDatasetFilter(e.target.value as DatasetFilter)}
            >
              <option value="any">Any dataset (SST-2 or IMDB)</option>
              <option value="SST-2">SST-2 only</option>
              <option value="IMDB">IMDB only</option>
            </select>
          </div>
          <div className="cta-row">
            <button
              type="button"
              className="btn"
              disabled={sampling}
              onClick={() => loadSample(datasetFilter)}
            >
              {sampling ? "Loading…" : "New example"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setManualMode(true);
                setSampleMeta(null);
                setText("");
                setResult(null);
              }}
            >
              Enter manually
            </button>
          </div>
          <div>
            <label htmlFor="text">
              {manualMode ? "Manual input" : "Dataset example"}
              {sampleMeta && !manualMode && (
                <>
                  {" "}
                  · <span className="pill">{sampleMeta.dataset}</span>
                  {sampleMeta.gold_label_name && (
                    <>
                      {" "}
                      <span className="pill">
                        gold:{" "}
                        <span
                          className={
                            sampleMeta.gold_label_name === "positive"
                              ? "label-positive"
                              : "label-negative"
                          }
                        >
                          {sampleMeta.gold_label_name}
                        </span>
                      </span>
                    </>
                  )}
                </>
              )}
            </label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => onManualEdit(e.target.value)}
              placeholder="Type a sentence, or click New example…"
            />
          </div>
          <div>
            <label htmlFor="prep">Preprocessing (sklearn)</label>
            <select
              id="prep"
              value={preprocess}
              onChange={(e) => setPreprocess(e.target.value)}
            >
              <option value="article">Article pipeline (stopwords include negation)</option>
              <option value="negation">Negation-preserving stopword list</option>
            </select>
          </div>
          <button className="btn primary" disabled={loading || !text.trim()} onClick={onPredict}>
            {loading ? "Computing…" : "Run inference"}
          </button>
          {health && (
            <p className="muted">
              Device: {health.device}. Active registry: {health.active_run_id ?? "—"}.
              Loaded: {health.loaded_models.length ? health.loaded_models.join("; ") : "none"}.
            </p>
          )}
          {error && <p className="error">{error}</p>}
        </div>
        <div className="stack">
          <h2>Output · P(positive)</h2>
          {!result && (
            <p className="muted">No inference yet. Submit text to populate the table.</p>
          )}
          {result && sampleMeta?.gold_label_name && !manualMode && (
            <p className="muted">
              Dataset gold label:{" "}
              <span
                className={
                  sampleMeta.gold_label_name === "positive"
                    ? "label-positive"
                    : "label-negative"
                }
              >
                {sampleMeta.gold_label_name}
              </span>
              {" "}(for reference; not used by the models at inference time).
            </p>
          )}
          {result && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Label</th>
                    <th>P(pos)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.predictions).map(([name, pred]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        <span
                          className={
                            pred.label_name === "positive"
                              ? "label-positive"
                              : "label-negative"
                          }
                        >
                          {pred.label_name}
                        </span>
                      </td>
                      <td className="ppos-cell">
                        <div className="ppos-row">
                          <div className="bar-track" title={`${(pred.p_positive * 100).toFixed(1)}%`}>
                            <div
                              className="bar-fill"
                              style={{
                                width: `${Math.min(100, Math.max(0, pred.p_positive * 100))}%`,
                              }}
                            />
                          </div>
                          <span className="pred-label">{pred.p_positive.toFixed(3)}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
