export interface AutoFineTuneOutputResult {
  model_repo_url?: string;
  job_url?: string;
  eval_result?: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function stripTrailingUrlPunctuation(value: string): string {
  return value.trim().replace(/[.,;:!?]+$/, '');
}

function isHuggingFaceUrl(value: string): URL | null {
  try {
    const url = new URL(stripTrailingUrlPunctuation(value));
    return url.origin === 'https://huggingface.co' ? url : null;
  } catch {
    return null;
  }
}

export function sanitizeAutoFineTuneJobUrl(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const url = isHuggingFaceUrl(value);
  return url?.pathname.startsWith('/jobs/') ? url.toString() : undefined;
}

export function sanitizeAutoFineTuneModelUrl(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const url = isHuggingFaceUrl(value);
  if (!url) return undefined;

  const [owner, repo] = url.pathname.split('/').filter(Boolean);
  if (!owner || !repo) return undefined;

  const reservedNamespaces = new Set([
    'api',
    'collections',
    'datasets',
    'docs',
    'jobs',
    'models',
    'organizations',
    'papers',
    'pricing',
    'settings',
    'spaces',
  ]);
  return reservedNamespaces.has(owner) ? undefined : url.toString();
}

function extractExplicitModelUrl(output: string): string | undefined {
  const match = output.match(/(?:^|\s)AUTO_FINETUNE_MODEL_URL\s*=\s*["']?(https:\/\/huggingface\.co\/[^\s"'`)]+)/m);
  return sanitizeAutoFineTuneModelUrl(match?.[1]);
}

function extractJobUrl(output: string): string | undefined {
  const match = output.match(/https:\/\/huggingface\.co\/jobs\/[^\s)`]+/);
  return sanitizeAutoFineTuneJobUrl(match?.[0]);
}

export function autoFineTuneResultFromOutput(output: string | undefined): AutoFineTuneOutputResult {
  if (!output) return {};
  try {
    const parsed = JSON.parse(output);
    const record = asRecord(parsed);
    if (!record) return {};
    return {
      model_repo_url: sanitizeAutoFineTuneModelUrl(asString(record.model_repo_url) ?? asString(record.model_url)),
      job_url: sanitizeAutoFineTuneJobUrl(asString(record.job_url)),
      eval_result: asString(record.eval_result),
    };
  } catch {
    return {
      model_repo_url: extractExplicitModelUrl(output),
      job_url: extractJobUrl(output),
    };
  }
}
