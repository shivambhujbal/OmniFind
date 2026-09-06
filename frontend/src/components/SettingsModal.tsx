import { useState, useEffect } from 'react';
import { X, CheckCircle, XCircle, Cpu, HardDrive, Shield } from 'lucide-react';
import { getHealthDetail, BackendError } from '../api/client';
import type { HealthDetailResponse } from '../api/types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [detail, setDetail] = useState<HealthDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    getHealthDetail()
      .then((d) => {
        setDetail(d);
        setLoading(false);
      })
      .catch((e) => {
        setError(e instanceof BackendError ? e.message : String(e));
        setLoading(false);
      });
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">System & Settings</div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        {loading && (
          <div style={{ textAlign: 'center', padding: '32px 0', color: 'var(--text-muted)' }}>
            Loading settings...
          </div>
        )}

        {error && <div className="alert alert-error">{error}</div>}

        {detail && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* System Info */}
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={16} color="var(--primary)" />
                Machine Learning & Runtime
              </h3>
              <table className="data-table">
                <tbody>
                  <tr>
                    <td><strong>ML Device</strong></td>
                    <td>{detail.ml_device}</td>
                  </tr>
                  <tr>
                    <td><strong>Python Runtime</strong></td>
                    <td>{detail.python}</td>
                  </tr>
                  <tr>
                    <td><strong>Platform</strong></td>
                    <td>{detail.platform}</td>
                  </tr>
                  <tr>
                    <td><strong>OmniFind Version</strong></td>
                    <td>v{detail.version}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Offline Environment */}
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Shield size={16} color="var(--primary)" />
                Air-Gapped Offline Protection
              </h3>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                All flags must be confirmed to ensure 100% private, local processing.
              </p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(detail.offline_env).map(([key, val]) => (
                    <tr key={key}>
                      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{key}</td>
                      <td>
                        {val === '<unset>' ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--warning)', fontSize: '12px' }}>
                            <XCircle size={14} /> Unset
                          </span>
                        ) : (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--success)', fontSize: '12px' }}>
                            <CheckCircle size={14} /> {val}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Data Locations */}
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <HardDrive size={16} color="var(--primary)" />
                Local Storage Locations
              </h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Store</th>
                    <th>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(detail.paths).map(([name, rep]) => (
                    <tr key={name}>
                      <td style={{ textTransform: 'capitalize' }}>{name.replace(/_/g, ' ')}</td>
                      <td>{rep.size_mb !== null ? `${rep.size_mb} MB` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
