import { useEffect, useState } from 'react';
import { X, Info, ShieldCheck, Zap, Database } from 'lucide-react';
import { getHealth, BackendError } from '../api/client';
import type { HealthResponse } from '../api/types';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AboutModal({ isOpen, onClose }: AboutModalProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    getHealth()
      .then(setHealth)
      .catch(() => {});
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">About OmniFind</div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ textAlign: 'center', padding: '16px 0' }}>
            <div
              style={{
                width: '60px',
                height: '60px',
                borderRadius: '16px',
                backgroundColor: 'var(--primary-light)',
                color: 'var(--primary)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '12px',
              }}
            >
              <Info size={32} />
            </div>
            <h2 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)' }}>
              OmniFind
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Local Semantic Search Utility
            </p>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <ShieldCheck size={20} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>100% Air-Gapped & Private</strong>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Your documents, notes, and photos never leave your machine. No telemetry or external cloud calls.
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <Zap size={20} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>Multimodal Semantic Search</strong>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Powered by local state-of-the-art vision & text embeddings (BGE-Large, OpenCLIP ViT-H/14, Moondream2).
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
              <Database size={20} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ fontSize: '13px', color: 'var(--text-primary)' }}>Instant Vector Retrieval</strong>
                <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Sub-second hybrid search combining dense vector embeddings with exact keyword matching.
                </p>
              </div>
            </div>
          </div>

          {health && (
            <div
              style={{
                backgroundColor: 'var(--neutral-light)',
                border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-md)',
                padding: '12px 16px',
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '12px',
                color: 'var(--text-muted)',
              }}
            >
              <span>Version: <strong>v{health.version}</strong></span>
              <span>Uptime: <strong>{Math.round(health.uptime_seconds)}s</strong></span>
              <span>Environment: <strong>Local Desktop</strong></span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
