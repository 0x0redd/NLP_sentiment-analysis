const steps = [
  { n: "2–3", title: "Data loading and exploratory analysis", api: "POST /api/runs/ (internal)" },
  { n: "4", title: "Article vs negation-aware preprocessing", api: "POST /api/preprocess/" },
  { n: "5–6", title: "TF-IDF and four article baselines", api: "Experiment artefacts" },
  { n: "7", title: "Feature-representation ablation", api: "GET /api/runs/{id}/ablation/" },
  { n: "8", title: "Contribution A — MLP on TF-IDF", api: "Included in trainer" },
  { n: "9", title: "Contribution B — DistilBERT fine-tune", api: "run_distilbert=true" },
  { n: "10", title: "Aggregate metrics and confusion matrices", api: "GET …/results/, …/confusion/" },
  { n: "10.2", title: "McNemar exact test with Holm correction", api: "GET …/mcnemar/" },
  { n: "11", title: "Controlled phrase inspection", api: "GET …/controlled/" },
  { n: "—", title: "Registry activation and inference", api: "POST …/activate/, /api/predict/" },
];

export default function PipelinePage() {
  return (
    <section className="panel stack">
      <div className="page-intro">
        <p className="section-label">Section · Methods</p>
        <h1>Correspondence between notebook and API</h1>
        <p>
          Each stage of <code>sentiment_merged.ipynb</code> is exposed through the
          Django service. The table below maps manuscript-style section numbers to
          endpoints used by this interface.
        </p>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>§</th>
              <th>Procedure</th>
              <th>Interface</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s) => (
              <tr key={s.n + s.title}>
                <td>{s.n}</td>
                <td>{s.title}</td>
                <td>
                  <code>{s.api}</code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
