import Link from "next/link";

export default function HomePage() {
  return (
    <section className="hero">
      <p className="kicker">Research interface · binary sentiment classification</p>
      <h1>Sentiment Analysis Based on Machine Learning Models</h1>
      <p>
        This application reproduces the experimental protocol of Liao (2024) and
        extends it with an MLP on TF-IDF and a DistilBERT fine-tune. Classical
        baselines (KNN, Random Forest, Naive Bayes, Logistic Regression) are trained
        on SST-2 and evaluated on held-out SST-2 and IMDB for cross-domain transfer.
      </p>
      <p>
        Use Inference to score controlled phrases under each model; Experiments to
        retrain; Results and Analysis for accuracy, confusion matrices, McNemar–Holm
        tests, and feature ablations.
      </p>
      <div className="cta-row">
        <Link className="btn primary" href="/predict">
          Open inference
        </Link>
        <Link className="btn" href="/results">
          View results
        </Link>
        <Link className="btn" href="/pipeline">
          Methods map
        </Link>
      </div>
      <p className="meta-line">
        Protocol: paper-faithful SST-2 splits · optional FAST_MODE subsample ·
        article vs negation-aware preprocessing
      </p>
    </section>
  );
}
