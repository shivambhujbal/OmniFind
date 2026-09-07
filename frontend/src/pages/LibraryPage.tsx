import { useState, useEffect, useRef, useCallback } from 'react';
import {
  FolderOpen,
  Upload,
  Trash2,
  ExternalLink,
  Eye,
  EyeOff,
  RotateCw,
  ChevronRight,
  FileText,
  Image as ImageIcon,
} from 'lucide-react';
import {
  scanPreview,
  scanFolder,
  uploadFiles,
  listFiles,
  getFile,
  getStats,
  deleteFile,
  reprocessFile,
  fileContentUrl,
  assetImageUrl,
  BackendError,
} from '../api/client';
import type {
  FileOut,
  FileDetailOut,
  LibraryStatsOut,
  ScanResponseOut,
} from '../api/types';
import Pagination from '../components/Pagination';

const STATUS_LABEL: Record<string, string> = {
  pending: 'queued',
  extracting: 'extracting',
  extracted: 'extracted',
  processing: 'processing',
  ready: 'ready',
  failed: 'failed',
};

const KIND_LABEL: Record<string, string> = {
  pdf: 'PDF',
  docx: 'DOCX',
  image: 'Image',
};

function humanSize(sizeBytes: number): string {
  const mb = sizeBytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(sizeBytes / 1024).toFixed(0)} KB`;
}

const PAGE_SIZE = 20;

// ── Folder Scanner ──────────────────────────────────────────────────

function FolderScanner({ onScanDone }: { onScanDone: () => void }) {
  const [path, setPath] = useState('');
  const [recursive, setRecursive] = useState(false);
  const [preview, setPreview] = useState<{ folder: string; count: number } | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResponseOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);

  // Debounced preview
  useEffect(() => {
    if (!path.trim()) { setPreview(null); return; }
    const timer = setTimeout(async () => {
      try {
        const p = await scanPreview(path, recursive);
        setPreview({ folder: p.folder, count: p.supported_files });
        setError(null);
      } catch (e) {
        setPreview(null);
        if (e instanceof BackendError) setError(e.message);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [path, recursive]);

  const handleScan = async () => {
    setScanning(true);
    setResult(null);
    setError(null);
    setProgress(0);
    try {
      const res = await scanFolder(path, recursive);
      setResult(res);
      if (res.queued > 0) {
        // Poll progress
        let peak = 0;
        const poll = setInterval(async () => {
          try {
            const stats = await getStats();
            const pending = stats.pending_jobs;
            peak = Math.max(peak, pending);
            if (pending === 0) {
              clearInterval(poll);
              setProgress(100);
              onScanDone();
            } else {
              const done = peak - pending;
              setProgress(Math.min(99, (done / peak) * 100));
            }
          } catch { /* ignore polling errors */ }
        }, 1500);
      } else {
        onScanDone();
      }
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  };

  return (
    <div>
      <h2 className="section-title">Add files</h2>
      <p className="caption" style={{ marginBottom: 'var(--space-md)' }}>
        Point the app at a folder on this machine. Your files are copied into
        the library and never moved or changed.
      </p>

      <div style={{ display: 'flex', gap: 'var(--space-sm)', marginBottom: 'var(--space-sm)' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <FolderOpen
            size={16}
            style={{
              position: 'absolute',
              left: 12,
              top: '50%',
              transform: 'translateY(-50%)',
              color: 'var(--text-tertiary)',
            }}
          />
          <input
            className="input"
            placeholder="C:\Users\you\Documents\Notes"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            style={{ paddingLeft: 36 }}
          />
        </div>
        <button
          className="btn btn-primary"
          disabled={!path || !preview?.count || scanning}
          onClick={handleScan}
        >
          {scanning ? 'Scanning…' : 'Index folder'}
        </button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={recursive}
            onChange={(e) => setRecursive(e.target.checked)}
          />
          Include subfolders
        </label>
        {preview && (
          <span className="caption">
            {preview.count} supported file(s) found in {preview.folder}
          </span>
        )}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="alert alert-success">
          {result.queued} file(s) added
          {result.already_present ? ` · ${result.already_present} already in the library` : ''}
          {result.unsupported ? ` · ${result.unsupported} unsupported` : ''}
          {result.too_large ? ` · ${result.too_large} over the size limit` : ''}
        </div>
      )}

      {progress !== null && progress < 100 && (
        <div>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="caption">Processing… {Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
}

// ── File Upload ─────────────────────────────────────────────────────

function FileUpload({ onUploadDone }: { onUploadDone: () => void }) {
  const [showUpload, setShowUpload] = useState(false);
  const [dragover, setDragover] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const handleFiles = async (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    if (!fileArray.length) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadFiles(fileArray);
      if (res.accepted > 0) {
        onUploadDone();
      }
      const rejected = res.results.filter((r) => !r.accepted);
      if (rejected.length) {
        setError(rejected.map((r) => `${r.original_name}: ${r.reason}`).join(', '));
      }
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ marginTop: 'var(--space-md)' }}>
      <button
        className="collapsible-header"
        onClick={() => setShowUpload(!showUpload)}
      >
        <ChevronRight
          size={14}
          style={{ transform: showUpload ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s' }}
        />
        Or upload individual files
      </button>

      {showUpload && (
        <div style={{ marginTop: 'var(--space-sm)' }}>
          <div
            className={`upload-zone${dragover ? ' dragover' : ''}`}
            onClick={() => fileInput.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
            onDragLeave={() => setDragover(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragover(false);
              if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
            }}
          >
            <Upload size={24} style={{ color: 'var(--text-tertiary)', marginBottom: 8 }} />
            <div className="upload-zone-text">
              {uploading ? 'Uploading…' : (
                <>Drop files here or <strong>browse</strong></>
              )}
            </div>
            <div className="caption" style={{ marginTop: 4 }}>
              PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP, WEBP
            </div>
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            style={{ display: 'none' }}
            accept=".pdf,.docx,.png,.jpg,.jpeg,.tiff,.bmp,.webp"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
          />
          {error && <div className="alert alert-error" style={{ marginTop: 8 }}>{error}</div>}
        </div>
      )}
    </div>
  );
}

// ── Stats Bar ───────────────────────────────────────────────────────

function StatsBar({ stats }: { stats: LibraryStatsOut }) {
  return (
    <div>
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{stats.total_files.toLocaleString()}</div>
          <div className="metric-label">Files</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{stats.total_chunks.toLocaleString()}</div>
          <div className="metric-label">Text passages</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{stats.total_assets.toLocaleString()}</div>
          <div className="metric-label">Images</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{stats.pending_jobs}</div>
          <div className="metric-label">Jobs running</div>
        </div>
      </div>
      <div className="caption">
        Disk — originals {stats.disk_usage_mb.originals} MB · extracted{' '}
        {stats.disk_usage_mb.derived} MB · vectors {stats.disk_usage_mb.vectors} MB ·
        models {stats.disk_usage_mb.models} MB
      </div>
    </div>
  );
}

// ── File Detail ─────────────────────────────────────────────────────

function FileDetail({ fileId }: { fileId: number }) {
  const [detail, setDetail] = useState<FileDetailOut | null>(null);
  const [tab, setTab] = useState<'text' | 'images'>('text');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getFile(fileId)
      .then(setDetail)
      .catch((e) => setError(e instanceof BackendError ? e.message : String(e)));
  }, [fileId]);

  if (error) return <div className="alert alert-error">{error}</div>;
  if (!detail) return <div className="loading-state"><div className="spinner" /></div>;

  return (
    <div className="file-detail fade-in">
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="metric-card">
          <div className="metric-value" style={{ fontSize: 16 }}>
            {STATUS_LABEL[detail.status] ?? detail.status}
          </div>
          <div className="metric-label">Status</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ fontSize: 16 }}>{detail.chunk_count}</div>
          <div className="metric-label">Passages</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ fontSize: 16 }}>{detail.asset_count}</div>
          <div className="metric-label">Images</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ fontSize: 16 }}>
            {detail.page_count ?? '—'}
          </div>
          <div className="metric-label">Pages</div>
        </div>
      </div>

      {detail.error && (
        <div className="alert alert-error">Extraction failed: {detail.error}</div>
      )}

      <div className="detail-tabs">
        <button
          className={`detail-tab${tab === 'text' ? ' active' : ''}`}
          onClick={() => setTab('text')}
        >
          <FileText size={14} /> Text ({detail.chunk_count})
        </button>
        <button
          className={`detail-tab${tab === 'images' ? ' active' : ''}`}
          onClick={() => setTab('images')}
        >
          <ImageIcon size={14} /> Images ({detail.asset_count})
        </button>
      </div>

      {tab === 'text' && (
        <div className="chunk-list">
          {detail.chunks.length === 0 ? (
            <div className="caption">No extractable text.</div>
          ) : (
            detail.chunks.slice(0, 40).map((chunk) => (
              <div key={chunk.id} className="chunk-item">
                <div className="chunk-meta">
                  #{chunk.ordinal} · {chunk.page_number ? `page ${chunk.page_number}` : '—'} ·{' '}
                  {chunk.kind}
                </div>
                <div className="chunk-text">
                  {chunk.text.length > 600
                    ? chunk.text.substring(0, 600) + '…'
                    : chunk.text}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'images' && (
        <div className="assets-grid">
          {detail.assets
            .filter((a) => a.kind !== 'thumbnail')
            .slice(0, 12)
            .map((asset) => (
              <div key={asset.id} className="asset-card">
                <img
                  src={assetImageUrl(fileId, asset.id)}
                  alt={asset.caption ?? ''}
                  loading="lazy"
                />
                <div className="asset-card-info">
                  {asset.kind.replace(/_/g, ' ')}
                  {asset.page_number ? ` · page ${asset.page_number}` : ''}
                  {asset.caption && (
                    <div style={{ fontStyle: 'italic', marginTop: 2 }}>
                      "{asset.caption}"
                    </div>
                  )}
                </div>
              </div>
            ))}
          {detail.assets.filter((a) => a.kind !== 'thumbnail').length === 0 && (
            <div className="caption">No images in this file.</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── File Row ────────────────────────────────────────────────────────

function FileRow({
  file,
  isOpen,
  onToggle,
  onDelete,
  onReprocess,
}: {
  file: FileOut;
  isOpen: boolean;
  onToggle: () => void;
  onDelete: () => void;
  onReprocess: () => void;
}) {
  const kind = KIND_LABEL[file.kind] ?? file.kind;
  const status = STATUS_LABEL[file.status] ?? file.status;
  const title = file.title || file.original_name;

  return (
    <div className="file-row">
      <div className="file-row-main">
        <div className="file-row-info">
          <div className="file-row-name">{title}</div>
          <div className="file-row-meta">
            {title !== file.original_name && `${file.original_name} — `}
            {kind} · {status} · {file.chunk_count} passages · {humanSize(file.size_bytes)}
          </div>
        </div>
        <div className="file-row-actions">
          <button className="btn btn-ghost btn-sm" onClick={onToggle}>
            {isOpen ? <EyeOff size={14} /> : <Eye size={14} />}
            {isOpen ? 'Hide' : 'Details'}
          </button>
          <a
            href={fileContentUrl(file.id)}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost btn-sm"
          >
            <ExternalLink size={14} /> Open
          </a>
          <button className="btn btn-ghost btn-sm" onClick={onReprocess} title="Reprocess">
            <RotateCw size={14} />
          </button>
          <button className="btn btn-danger btn-sm" onClick={onDelete}>
            <Trash2 size={14} /> Remove
          </button>
        </div>
      </div>
      {isOpen && <FileDetail fileId={file.id} />}
    </div>
  );
}

// ── Main Library Page ───────────────────────────────────────────────

export default function LibraryPage() {
  const [files, setFiles] = useState<FileOut[]>([]);
  const [stats, setStats] = useState<LibraryStatsOut | null>(null);
  const [filter, setFilter] = useState('');
  const [page, setPage] = useState(1);
  const [openFileId, setOpenFileId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [f, s] = await Promise.all([listFiles(1000), getStats()]);
      setFiles(f);
      setStats(s);
      setError(null);
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const filtered = filter
    ? files.filter(
        (f) =>
          f.original_name.toLowerCase().includes(filter.toLowerCase()) ||
          (f.title ?? '').toLowerCase().includes(filter.toLowerCase()),
      )
    : files;

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const visible = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const handleDelete = async (fileId: number) => {
    try {
      await deleteFile(fileId);
      if (openFileId === fileId) setOpenFileId(null);
      loadData();
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    }
  };

  const handleReprocess = async (fileId: number) => {
    try {
      await reprocessFile(fileId);
      loadData();
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    }
  };

  return (
    <div className="page-container fade-in">
      <h1 className="page-title">Library</h1>
      <p className="page-subtitle">
        Manage your indexed files. Point the app at a folder or upload files directly.
      </p>

      {error && <div className="alert alert-error">{error}</div>}

      <FolderScanner onScanDone={loadData} />
      <FileUpload onUploadDone={loadData} />

      <hr className="divider" />

      {stats && <StatsBar stats={stats} />}

      <hr className="divider" />

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 'var(--space-md)' }}>
        <h2 className="section-title" style={{ marginBottom: 0 }}>
          Library ({filtered.length})
        </h2>
        <input
          className="input"
          style={{ maxWidth: 280 }}
          placeholder="Filter by name…"
          value={filter}
          onChange={(e) => { setFilter(e.target.value); setPage(1); }}
        />
      </div>

      {visible.length === 0 ? (
        <div className="empty-state">
          <FolderOpen />
          <div className="empty-state-title">
            {files.length === 0 ? 'Nothing in the library yet' : 'No files match that name'}
          </div>
          <div className="empty-state-text">
            {files.length === 0
              ? 'Add a folder above to get started.'
              : 'Try a different search term.'}
          </div>
        </div>
      ) : (
        <div className="file-list">
          {visible.map((file) => (
            <FileRow
              key={file.id}
              file={file}
              isOpen={openFileId === file.id}
              onToggle={() => setOpenFileId(openFileId === file.id ? null : file.id)}
              onDelete={() => handleDelete(file.id)}
              onReprocess={() => handleReprocess(file.id)}
            />
          ))}
        </div>
      )}

      <Pagination
        page={currentPage}
        totalPages={totalPages}
        total={filtered.length}
        pageSize={PAGE_SIZE}
        onPageChange={setPage}
      />
    </div>
  );
}
