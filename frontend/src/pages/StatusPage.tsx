import { useEffect, useState, useRef } from 'react';
import {
  RotateCw,
  FolderOpen,
  FileText,
  Image as ImageIcon,
  CheckCircle,
  AlertCircle,
  Cpu,
  Shield,
  Layers,
} from 'lucide-react';
import {
  getStats,
  scanPreview,
  scanFolder,
  getHealthDetail,
  BackendError,
} from '../api/client';
import type { LibraryStatsOut, HealthDetailResponse, ScanResponseOut } from '../api/types';

export default function StatusPage() {
  const [stats, setStats] = useState<LibraryStatsOut | null>(null);
  const [health, setHealth] = useState<HealthDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Folder scanning state
  const [showFolderInput, setShowFolderInput] = useState(false);
  const [folderPath, setFolderPath] = useState('');
  const [recursive, setRecursive] = useState(false);
  const [previewCount, setPreviewCount] = useState<number | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<ScanResponseOut | null>(null);
  const [currentFileName, setCurrentFileName] = useState<string>('DBMS_Notes.pdf');

  const folderInputRef = useRef<HTMLInputElement>(null);

  const fetchStats = async () => {
    try {
      const [s, h] = await Promise.all([getStats(), getHealthDetail()]);
      setStats(s);
      setHealth(h);
      setError(null);
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 3000);
    return () => clearInterval(interval);
  }, []);

  // Debounced scan preview
  useEffect(() => {
    if (!folderPath.trim()) {
      setPreviewCount(null);
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const preview = await scanPreview(folderPath, recursive);
        setPreviewCount(preview.supported_files);
      } catch {
        setPreviewCount(null);
      }
    }, 400);
    return () => clearTimeout(timer);
  }, [folderPath, recursive]);

  const handleStartScan = async () => {
    if (!folderPath.trim()) return;
    setIsScanning(true);
    setScanResult(null);
    setError(null);
    try {
      const res = await scanFolder(folderPath, recursive);
      setScanResult(res);
      fetchStats();
    } catch (e) {
      setError(e instanceof BackendError ? e.message : String(e));
    } finally {
      setIsScanning(false);
    }
  };

  const isIndexingActive = (stats && stats.pending_jobs > 0) || isScanning;
  const totalFiles = stats?.total_files ?? 0;
  const pendingJobs = stats?.pending_jobs ?? 0;
  const processedDocs = Math.max(0, totalFiles - pendingJobs);
  const processedImages = stats?.total_assets ?? 0;
  const progressPct = totalFiles > 0 ? Math.min(100, Math.round((processedDocs / totalFiles) * 100)) : 0;

  let statusTitle = 'Index ready & up to date';
  if (isIndexingActive) {
    statusTitle = 'Indexing files...';
  } else if (totalFiles === 0) {
    statusTitle = 'No files indexed yet';
  }

  return (
    <div style={{ animation: 'fadeIn 0.2s ease-out' }}>
      <h1 className="page-title">Index Your Files</h1>
      <p className="page-subtitle">
        Select a folder to allow OmniFind to index supported files.
      </p>

      {error && <div className="alert alert-error">{error}</div>}

      {/* ── Indexing Status Card (matching style of Screenshot 1) ───── */}
      <div className="indexing-card">
        <div className="indexing-card-header">
          <div className="indexing-title-group">
            <RotateCw
              size={20}
              color="var(--primary)"
              className={isIndexingActive ? 'spin' : ''}
            />
            <span className="indexing-title">{statusTitle}</span>
          </div>
          <span className="indexing-count">
            {processedDocs} / {totalFiles} files
          </span>
        </div>

        {/* Progress Bar */}
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${isIndexingActive ? Math.max(progressPct, 5) : totalFiles > 0 ? 100 : 0}%` }}
          />
        </div>

        {/* Current File Row - displayed during active scanning / jobs */}
        {isIndexingActive && (
          <div className="current-file-row">
            <span>Current task:</span>
            <span className="current-file-badge">{pendingJobs > 0 ? `${pendingJobs} job(s) in queue` : 'Scanning folder...'}</span>
          </div>
        )}

        {/* Processed Counts Row */}
        <div className="processed-stats-row">
          <div className="processed-stat">
            <FileText size={16} color="var(--text-muted)" />
            <span>
              Documents processed: <strong>{processedDocs}</strong>
            </span>
          </div>

          <div className="processed-stat">
            <ImageIcon size={16} color="var(--text-muted)" />
            <span>
              Images processed: <strong>{processedImages}</strong>
            </span>
          </div>
        </div>
      </div>

      {/* Select Folder Action Button (Screenshot 1) */}
      <div style={{ marginBottom: 'var(--space-xl)' }}>
        <button
          className="btn btn-outline"
          style={{ padding: '10px 22px', fontSize: '14px', gap: '10px' }}
          onClick={() => {
            setShowFolderInput(!showFolderInput);
            setTimeout(() => folderInputRef.current?.focus(), 100);
          }}
        >
          <FolderOpen size={18} color="var(--primary)" />
          Select Folder
        </button>
      </div>

      {/* Expandable Folder Scanner Panel */}
      {showFolderInput && (
        <div
          style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-lg)',
            padding: '24px',
            maxWidth: '680px',
            marginBottom: 'var(--space-xl)',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>
            Choose Local Directory to Scan
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
            Enter the absolute path of a folder containing PDFs, DOCX files, or images on your machine.
          </p>

          <div style={{ display: 'flex', gap: '10px', marginBottom: '12px' }}>
            <input
              ref={folderInputRef}
              className="search-pill-input"
              style={{
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 14px',
                fontSize: '13px',
                backgroundColor: 'var(--bg-input)',
              }}
              placeholder="e.g. C:\Users\YourName\Documents"
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
            />
            <button
              className="btn btn-primary"
              disabled={!folderPath || isScanning}
              onClick={handleStartScan}
            >
              {isScanning ? 'Scanning...' : 'Start Indexing'}
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-muted)' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
              />
              Scan subfolders recursively
            </label>

            {previewCount !== null && (
              <span style={{ color: 'var(--primary)', fontWeight: 500 }}>
                Found {previewCount} supported files
              </span>
            )}
          </div>

          {scanResult && (
            <div className="alert alert-info" style={{ marginTop: '16px', marginBottom: 0 }}>
              <CheckCircle size={16} />
              <span>
                Scanned {scanResult.found} files ({scanResult.queued} queued, {scanResult.skipped} skipped).
              </span>
            </div>
          )}
        </div>
      )}

      {/* Backend Diagnostics & Info */}
      {health && (
        <div style={{ maxWidth: '680px', marginTop: 'var(--space-2xl)' }}>
          <h2 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '12px' }}>
            System Environment
          </h2>
          <table className="data-table">
            <tbody>
              <tr>
                <td><strong>ML Device</strong></td>
                <td>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                    <Cpu size={14} color="var(--primary)" />
                    {health.ml_device}
                  </span>
                </td>
              </tr>
              <tr>
                <td><strong>Air-Gapped Offline Protection</strong></td>
                <td>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--success)' }}>
                    <Shield size={14} /> Active (Zero Cloud Calls)
                  </span>
                </td>
              </tr>
              <tr>
                <td><strong>Vector Database</strong></td>
                <td>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                    <Layers size={14} color="var(--primary)" />
                    ChromaDB (Local Persistent)
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
