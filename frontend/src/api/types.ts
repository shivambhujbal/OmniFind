/* ── API response types ─────────────────────────────────────────────────
 *  Mirror of the Pydantic schemas in sidecar/app/api/schemas.py and
 *  sidecar/app/api/search.py. The field names match the JSON wire format
 *  exactly so no mapping layer is needed.
 * ─────────────────────────────────────────────────────────────────────── */

// --- Health ---------------------------------------------------------------

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface PathReport {
  path: string;
  exists: boolean;
  size_mb: number | null;
}

export interface HealthDetailResponse {
  status: string;
  version: string;
  uptime_seconds: number;
  python: string;
  platform: string;
  ml_device: string;
  offline_env: Record<string, string>;
  paths: Record<string, PathReport>;
}

// --- Files ----------------------------------------------------------------

export type FileKind = 'pdf' | 'docx' | 'image';
export type ProcessingStatus =
  | 'pending'
  | 'extracting'
  | 'extracted'
  | 'processing'
  | 'ready'
  | 'failed';

export interface FileOut {
  id: number;
  sha256: string;
  original_name: string;
  title: string | null;
  kind: FileKind;
  mime_type: string;
  size_bytes: number;
  page_count: number | null;
  status: ProcessingStatus;
  error: string | null;
  created_at: string;
  updated_at: string;
  chunk_count: number;
  asset_count: number;
}

export interface ChunkOut {
  id: number;
  kind: string;
  ordinal: number;
  page_number: number | null;
  text: string;
  char_count: number;
}

export interface AssetOut {
  id: number;
  kind: string;
  page_number: number | null;
  width: number | null;
  height: number | null;
  caption: string | null;
}

export interface FileDetailOut extends FileOut {
  chunks: ChunkOut[];
  assets: AssetOut[];
}

export interface UploadResultOut {
  original_name: string;
  accepted: boolean;
  duplicate: boolean;
  file_id: number | null;
  job_id: string | null;
  reason: string | null;
}

export interface UploadResponse {
  results: UploadResultOut[];
  accepted: number;
  rejected: number;
}

export interface ScanPreviewOut {
  folder: string;
  supported_files: number;
}

export interface ScanResponseOut {
  folder: string;
  queued: number;
  already_present: number;
  unsupported: number;
  too_large: number;
  failed: string[];
}

export interface LibraryStatsOut {
  total_files: number;
  by_status: Record<string, number>;
  by_kind: Record<string, number>;
  total_chunks: number;
  total_assets: number;
  pending_jobs: number;
  disk_usage_mb: Record<string, number>;
}

// --- Search ---------------------------------------------------------------

export interface SearchResultOut {
  file_id: number;
  file_name: string;
  file_kind: string;
  title: string | null;
  score: number;
  similarity: number;
  match_kind: string;
  page_number: number | null;
  snippet: string | null;
  chunk_id: number | null;
  asset_id: number | null;
  exact: boolean;
  supporting: SearchResultOut[];
}

export interface SearchResponseOut {
  query: string;
  results: SearchResultOut[];
  text_candidates: number;
  image_candidates: number;
  searched_images: boolean;
  confident: boolean;
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface IndexStatusOut {
  collections: Record<string, number>;
  text_dim: number;
  image_dim: number;
  image_search: boolean;
}

// --- Jobs -----------------------------------------------------------------

export interface JobOut {
  id: string;
  name: string;
  state: string;
  detail: string;
  file_id: number | null;
  error: string | null;
  priority: number;
  queued_seconds: number;
  duration_seconds: number | null;
}

// --- Models ---------------------------------------------------------------

export interface ModelsResponse {
  [key: string]: unknown;
}
