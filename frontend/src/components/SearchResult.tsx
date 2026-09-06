import { useState } from 'react';
import {
  FileText,
  Image as ImageIcon,
  Code,
  Folder,
  File,
  Calendar,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { assetImageUrl, fileContentUrl } from '../api/client';
import type { SearchResultOut } from '../api/types';

interface SearchResultProps {
  result: SearchResultOut;
  rank: number;
}

export default function SearchResult({ result }: SearchResultProps) {
  const [showSupporting, setShowSupporting] = useState(false);

  // Compute relevance percentage (e.g., 94% relevant)
  const relevancePct = Math.min(99, Math.round(result.similarity * 100));

  // Determine file type and icon
  const isCode =
    result.file_name.endsWith('.sql') ||
    result.file_name.endsWith('.py') ||
    result.file_name.endsWith('.js') ||
    result.file_name.endsWith('.ts') ||
    result.file_name.endsWith('.json');

  const isImage =
    result.file_kind === 'image' ||
    result.match_kind === 'embedded_image' ||
    result.match_kind === 'source_image' ||
    result.match_kind === 'page_image';

  // Build breadcrumb folder path
  const folderPath = result.file_path
    ? result.file_path.replace(/\\/g, ' / ')
    : 'Documents / Library';

  // Extract tags from caption or metadata
  const tags: string[] = [];
  if (isImage && result.snippet) {
    // Extract keywords from caption
    const words = result.snippet
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length > 3 && !['with', 'from', 'that', 'this', 'have', 'were'].includes(w));
    tags.push(...Array.from(new Set(words)).slice(0, 4));
  }

  return (
    <div className="result-card">
      {/* Header */}
      <div className="result-header">
        <div className="result-title-group">
          <div className="file-type-icon">
            {isCode ? (
              <Code size={18} color="var(--primary)" />
            ) : isImage ? (
              <ImageIcon size={18} color="#bc4800" />
            ) : (
              <FileText size={18} color="var(--primary)" />
            )}
          </div>
          <span className="result-filename">{result.file_name}</span>
          <span className="relevance-pill">
            {relevancePct > 0 ? `${relevancePct}% relevant` : 'Matched'}
          </span>
        </div>

        <a
          href={fileContentUrl(result.file_id)}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-primary"
        >
          Open File
        </a>
      </div>

      {/* Body / Snippet */}
      <div className="result-body">
        {/* Image Thumbnail if image asset available */}
        {result.asset_id && (
          <div className="result-thumbnail-wrapper">
            <img
              className="result-thumbnail"
              src={assetImageUrl(result.file_id, result.asset_id)}
              alt={result.file_name}
              loading="lazy"
            />
          </div>
        )}

        <div className="result-content-col">
          {result.snippet && (
            isCode ? (
              <pre className="code-snippet">{result.snippet}</pre>
            ) : (
              <p className="result-snippet">{result.snippet}</p>
            )
          )}

          {/* Tag pills */}
          {tags.length > 0 && (
            <div className="tags-row">
              {tags.map((tag) => (
                <span key={tag} className="tag-chip">
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer Metadata */}
      <div className="result-footer">
        <div className="meta-item">
          <Folder size={14} />
          <span>{folderPath}</span>
        </div>

        {result.page_number && (
          <div className="meta-item">
            <File size={14} />
            <span>Page: {result.page_number}</span>
          </div>
        )}

        {result.page_number ? (
          <div className="meta-item">
            <File size={14} />
            <span>Page: {result.page_number}</span>
          </div>
        ) : null}

        {result.match_kind ? (
          <div className="meta-item">
            <span>Type: {result.match_kind.replace(/_/g, ' ')}</span>
          </div>
        ) : null}
      </div>

      {/* Supporting Matches Toggle */}
      {result.supporting && result.supporting.length > 0 && (
        <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-subtle)', paddingTop: '8px' }}>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '11px', padding: '4px 10px' }}
            onClick={() => setShowSupporting(!showSupporting)}
          >
            {showSupporting ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {result.supporting.length} more match(es) in this file
          </button>

          {showSupporting && (
            <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {result.supporting.map((extra, i) => (
                <div
                  key={i}
                  style={{
                    backgroundColor: 'var(--neutral-light)',
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '12px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                    {extra.match_kind} {extra.page_number ? `· Page ${extra.page_number}` : ''}
                  </div>
                  {extra.snippet && <div>{extra.snippet}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
