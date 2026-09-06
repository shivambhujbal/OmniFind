/* ── API client ─────────────────────────────────────────────────────────
 *  Fetch-based HTTP client for the FastAPI sidecar. Mirrors the Python
 *  ApiClient in streamlit_app/api_client.py exactly — same endpoints,
 *  same parameter names — so the contract is tested by both frontends.
 *
 *  In dev mode, Vite proxies /api/* → http://127.0.0.1:8756/*
 *  In production (Tauri), the base URL is the loopback port directly.
 * ─────────────────────────────────────────────────────────────────────── */

import type {
  FileDetailOut,
  FileOut,
  HealthDetailResponse,
  HealthResponse,
  IndexStatusOut,
  JobOut,
  LibraryStatsOut,
  ModelsResponse,
  ScanPreviewOut,
  ScanResponseOut,
  SearchResponseOut,
  UploadResponse,
} from './types';

const BASE_URL = '/api';

export class BackendError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'BackendError';
    this.status = status;
  }
}

async function request<T>(
  method: string,
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  let response: Response;

  try {
    response = await fetch(url, { method, ...options });
  } catch {
    throw new BackendError(
      `Cannot reach the backend. Is the sidecar running?`,
      0,
    );
  }

  if (!response.ok) {
    let detail: string;
    try {
      const body = await response.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new BackendError(`${response.status}: ${detail}`, response.status);
  }

  return response.json() as Promise<T>;
}

function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  let url = path;
  if (params) {
    const search = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      search.set(key, String(value));
    }
    url += `?${search.toString()}`;
  }
  return request<T>('GET', url);
}

// ── Health ───────────────────────────────────────────────────────────

export function getHealth(): Promise<HealthResponse> {
  return get<HealthResponse>('/health');
}

export function getHealthDetail(): Promise<HealthDetailResponse> {
  return get<HealthDetailResponse>('/health/detail');
}

// ── Files ────────────────────────────────────────────────────────────

export function uploadFiles(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  return request<UploadResponse>('POST', '/files', { body: formData });
}

export function scanPreview(
  path: string,
  recursive = false,
): Promise<ScanPreviewOut> {
  return get<ScanPreviewOut>('/files/scan/preview', { path, recursive });
}

export function scanFolder(
  path: string,
  recursive = false,
): Promise<ScanResponseOut> {
  return request<ScanResponseOut>('POST', '/files/scan', {
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, recursive }),
  });
}

export function listFiles(limit = 200): Promise<FileOut[]> {
  return get<FileOut[]>('/files', { limit });
}

export function getFile(fileId: number): Promise<FileDetailOut> {
  return get<FileDetailOut>(`/files/${fileId}`);
}

export function deleteFile(fileId: number): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>('DELETE', `/files/${fileId}`);
}

export function reprocessFile(fileId: number): Promise<JobOut> {
  return request<JobOut>('POST', `/files/${fileId}/reprocess`);
}

export function getStats(): Promise<LibraryStatsOut> {
  return get<LibraryStatsOut>('/files/stats');
}

export function getSupportedTypes(): Promise<string[]> {
  return get<string[]>('/files/supported-types');
}

export function fileContentUrl(fileId: number): string {
  return `${BASE_URL}/files/${fileId}/content`;
}

export function assetImageUrl(fileId: number, assetId: number): string {
  return `${BASE_URL}/files/${fileId}/assets/${assetId}/image`;
}

// ── Search ───────────────────────────────────────────────────────────

export function search(params: {
  q: string;
  limit?: number;
  include_images?: boolean;
  group_by_file?: boolean;
  kinds?: string;
  page?: number;
  page_size?: number;
}): Promise<SearchResponseOut> {
  return get<SearchResponseOut>('/search', params as Record<string, string | number | boolean>);
}

export function getSearchStatus(): Promise<IndexStatusOut> {
  return get<IndexStatusOut>('/search/status');
}

// ── Models ───────────────────────────────────────────────────────────

export function getModels(): Promise<ModelsResponse> {
  return get<ModelsResponse>('/models');
}

// ── Jobs ─────────────────────────────────────────────────────────────

export function getJobs(limit = 25): Promise<JobOut[]> {
  return get<JobOut[]>('/jobs', { limit });
}
