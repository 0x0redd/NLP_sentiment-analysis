const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    cache: "no-store",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP ${res.status}`);
  }
  return data as T;
}

export type Health = {
  ok: boolean;
  device: string;
  training: boolean;
  loaded_models: string[];
  active_run_id: string | null;
  last_succeeded_run_id: string | null;
};

export type ExperimentRun = {
  id: string;
  status: string;
  options: Record<string, boolean>;
  progress: string;
  error: string;
  summary: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type PredictResult = {
  text: string;
  preprocess: string;
  predictions: Record<
    string,
    { label: number; label_name: string; p_positive: number }
  >;
};

export type SampleText = {
  dataset: string;
  text: string;
  source: string;
  gold_label?: number;
  gold_label_name?: string;
};

export const api = {
  health: () => request<Health>("/api/health/"),
  listRuns: () => request<ExperimentRun[]>("/api/runs/"),
  getRun: (id: string) => request<ExperimentRun>(`/api/runs/${id}/`),
  createRun: (body: {
    fast_mode?: boolean;
    run_distilbert?: boolean;
    run_ablation?: boolean;
    run_negation?: boolean;
  }) =>
    request<ExperimentRun>("/api/runs/", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  activate: (id: string) =>
    request<{ active_run_id: string; loaded_models: string[] }>(
      `/api/runs/${id}/activate/`,
      { method: "POST", body: "{}" }
    ),
  results: (id: string) =>
    request<{ run_id: string; results: Record<string, unknown>[] }>(
      `/api/runs/${id}/results/`
    ),
  mcnemar: (id: string) =>
    request<{ run_id: string; mcnemar: Record<string, unknown>[] }>(
      `/api/runs/${id}/mcnemar/`
    ),
  ablation: (id: string) =>
    request<{ run_id: string; ablation: Record<string, unknown>[] }>(
      `/api/runs/${id}/ablation/`
    ),
  confusion: (id: string) =>
    request<{ run_id: string; confusion: Record<string, Record<string, number[][]>> }>(
      `/api/runs/${id}/confusion/`
    ),
  timings: (id: string) =>
    request<{ run_id: string; timings: Record<string, number> }>(
      `/api/runs/${id}/timings/`
    ),
  controlled: (id: string) =>
    request<{ run_id: string; controlled: Record<string, unknown>[] }>(
      `/api/runs/${id}/controlled/`
    ),
  predict: (body: { text: string; models?: string[]; preprocess?: string }) =>
    request<PredictResult>("/api/predict/", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  preprocess: (text: string) =>
    request<{ text: string; article: string; negation: string }>(
      "/api/preprocess/",
      { method: "POST", body: JSON.stringify({ text }) }
    ),
  sample: (dataset?: "SST-2" | "IMDB" | "any") => {
    const q = dataset && dataset !== "any" ? `?dataset=${encodeURIComponent(dataset)}` : "";
    return request<SampleText>(`/api/samples/${q}`);
  },
};
