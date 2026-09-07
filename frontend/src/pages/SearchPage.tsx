import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Search as SearchIcon,
  X,
  FolderOpen,
  AlertTriangle,
  FileQuestion,
} from 'lucide-react';
import {
  search as searchApi,
  getSearchStatus,
  BackendError,
} from '../api/client';
import type { SearchResponseOut, IndexStatusOut } from '../api/types';
import SearchResult from '../components/SearchResult';
import Pagination from '../components/Pagination';

const KIND_TABS = [
  { label: 'All', value: 'all' },
  { label: 'Documents', value: 'documents' },
  { label: 'Images', value: 'images' },
] as const;

function saveSearchHistory(query: string, resultCount: number) {
  try {
    const raw = localStorage.getItem('omnifind_search_history');
    const list = raw ? JSON.parse(raw) : [];
    const filtered = list.filter((item: { query: string }) => item.query.toLowerCase() !== query.toLowerCase());
    filtered.unshift({
      id: String(Date.now()),
      query,
      timestamp: 'Just now',
      count: resultCount,
    });
    localStorage.setItem('omnifind_search_history', JSON.stringify(filtered.slice(0, 20)));
  } catch {
    // Ignore storage errors
  }
}

export default function SearchPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const initialQuery = searchParams.get('q') || '';

  const [query, setQuery] = useState(initialQuery);
  const [activeTab, setActiveTab] = useState<string>('all');
  const [page, setPage] = useState(1);
  const [pageSize] = useState(10);
  const [groupByFile] = useState(true);

  const [status, setStatus] = useState<IndexStatusOut | null>(null);
  const [response, setResponse] = useState<SearchResponseOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(Boolean(initialQuery));

  const inputRef = useRef<HTMLInputElement>(null);

  // Load index status once
  useEffect(() => {
    getSearchStatus()
      .then(setStatus)
      .catch((e) => setError(e instanceof BackendError ? e.message : String(e)));
  }, []);

  const doSearch = useCallback(
    async (q: string, p: number, tab: string) => {
      if (!q.trim()) {
        setResponse(null);
        setHasSearched(false);
        return;
      }
      setLoading(true);
      setError(null);
      setHasSearched(true);
      try {
        const imagesOn = status?.image_search ?? true;
        const res = await searchApi({
          q: q.trim(),
          include_images: imagesOn,
          group_by_file: groupByFile,
          kinds: tab,
          page: p,
          page_size: pageSize,
        });
        setResponse(res);
        saveSearchHistory(q.trim(), res.total);
      } catch (e) {
        setError(e instanceof BackendError ? e.message : String(e));
        setResponse(null);
      } finally {
        setLoading(false);
      }
    },
    [groupByFile, pageSize, status],
  );

  // Sync if URL search param changes
  useEffect(() => {
    const q = searchParams.get('q');
    if (q) {
      setQuery(q);
      doSearch(q, 1, activeTab);
    }
  }, [searchParams]);

  // Global Ctrl+K shortcut to focus search input
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Re-search when tab changes
  const handleTabClick = (tabValue: string) => {
    setActiveTab(tabValue);
    setPage(1);
    if (query.trim()) {
      doSearch(query, 1, tabValue);
    }
  };

  // Handle submit
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    doSearch(query, 1, activeTab);
  };

  const handleClear = () => {
    setQuery('');
    setResponse(null);
    setHasSearched(false);
    inputRef.current?.focus();
  };

  const totalIndexedVectors = status
    ? Object.values(status.collections).reduce((a, b) => a + b, 0)
    : 0;

  // 1. SEARCH RESULTS VIEW (when search was executed or response exists)
  if (hasSearched || response) {
    return (
      <div className="page-container" style={{ animation: 'fadeIn 0.2s ease-out' }}>
        {/* Top Search Bar */}
        <form onSubmit={handleSubmit} style={{ maxWidth: '640px' }}>
          <div className="search-pill-bar">
            <SearchIcon size={18} color="var(--text-subtle)" />
            <input
              ref={inputRef}
              className="search-pill-input"
              placeholder="What are You Looking For......"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
            {query && (
              <button
                type="button"
                className="clear-btn"
                onClick={handleClear}
                aria-label="Clear search"
              >
                <X size={16} />
              </button>
            )}
            <span className="shortcut-badge">Ctrl + K</span>
          </div>
        </form>

        {/* Filter Tabs (All / Documents / Images) */}
        <div className="filter-tabs">
          {KIND_TABS.map((tab) => (
            <button
              key={tab.value}
              className={`filter-tab ${activeTab === tab.value ? 'active' : ''}`}
              onClick={() => handleTabClick(tab.value)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        {loading && (
          <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)' }}>
            <div className="spin" style={{ display: 'inline-block', marginBottom: '8px' }}>
              <SearchIcon size={24} color="var(--primary)" />
            </div>
            <div>Searching your files...</div>
          </div>
        )}

        {response && !loading && (
          <>
            {!response.confident && response.results.length > 0 && (
              <div className="alert alert-warning">
                <AlertTriangle size={16} />
                <span>Weak match — try describing the content in more detail.</span>
              </div>
            )}

            {response.results.length === 0 ? (
              <div className="empty-state-container" style={{ minHeight: '35vh' }}>
                <div className="empty-state-icon-box">
                  <FileQuestion size={32} />
                </div>
                <h3 className="empty-state-title">No matching files found</h3>
                <p className="empty-state-subtitle">
                  Nothing in your library closely matched "{query}". Try different terms, or index more files in the library.
                </p>
                <button className="btn btn-primary" onClick={() => navigate('/status')}>
                  <FolderOpen size={16} />
                  Index a Folder
                </button>
              </div>
            ) : (
              <div className="results-list">
                {response.results.map((res, i) => (
                  <SearchResult
                    key={`${res.file_id}-${res.chunk_id ?? res.asset_id ?? i}`}
                    result={res}
                    rank={(response.page - 1) * response.page_size + i + 1}
                  />
                ))}
              </div>
            )}

            {response.results.length > 0 && (
              <div className="results-status-footer">
                Showing {response.results.length} of {response.total} result(s) for "{query}".
              </div>
            )}

            {response.total_pages > 1 && (
              <Pagination
                page={response.page}
                totalPages={response.total_pages}
                total={response.total}
                pageSize={response.page_size}
                onPageChange={(p) => {
                  setPage(p);
                  doSearch(query, p, activeTab);
                }}
              />
            )}
          </>
        )}
      </div>
    );
  }

  // 2. HERO HOME VIEW
  return (
    <div className="hero-search-container">
      <h1 className="hero-title">OmniFind</h1>

      <form onSubmit={handleSubmit} style={{ width: '100%' }}>
        <div className="search-pill-bar">
          <SearchIcon size={18} color="var(--text-subtle)" />
          <input
            ref={inputRef}
            className="search-pill-input"
            placeholder="What are You Looking For......"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          {query && (
            <button
              type="button"
              className="clear-btn"
              onClick={handleClear}
              aria-label="Clear search"
            >
              <X size={16} />
            </button>
          )}
          <span className="shortcut-badge">Ctrl + K</span>
        </div>
      </form>

      {totalIndexedVectors === 0 && (
        <div style={{ marginTop: '32px' }}>
          <button
            className="btn btn-outline"
            style={{ fontSize: '13px', gap: '8px' }}
            onClick={() => navigate('/status')}
          >
            <FolderOpen size={15} color="var(--primary)" />
            Your files are not indexed yet — Select a folder to start
          </button>
        </div>
      )}
    </div>
  );
}
