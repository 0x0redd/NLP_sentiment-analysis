import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sentiment Analysis — Experimental Interface",
  description:
    "Reproduction of Liao (2024) with MLP and DistilBERT extensions. SST-2 / IMDB evaluation.",
};

const links = [
  { href: "/", label: "Abstract" },
  { href: "/predict", label: "Inference" },
  { href: "/train", label: "Experiments" },
  { href: "/results", label: "Results" },
  { href: "/compare", label: "Analysis" },
  { href: "/pipeline", label: "Methods" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <main>
          <header className="site-header">
            <div className="brand-block">
              <div className="brand">Sentiment Analysis</div>
              <p className="brand-sub">
                Liao (2024) reproduction · MLP · DistilBERT · PEM-SDIA / NLP
              </p>
            </div>
            <nav className="nav">
              {links.map((l) => (
                <Link key={l.href} href={l.href}>
                  {l.label}
                </Link>
              ))}
            </nav>
          </header>
          {children}
          <p className="footer-note">
            Binary polarity classification on SST-2 (in-domain) and IMDB (cross-domain).
            Models exported from the merged research notebook.
          </p>
        </main>
      </body>
    </html>
  );
}
