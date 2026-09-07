import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, Search, ArrowRight, Trash2 } from 'lucide-react';

interface HistoryItem {
  id: string;
  query: string;
  timestamp: string;
  count: number;
}

export default function SearchHistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    const saved = localStorage.getItem('omnifind_search_history');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return [];
      }
    }
    return [];
  });

  const clearHistory = () => {
    localStorage.removeItem('omnifind_search_history');
    setHistory([]);
  };

  const handleRunQuery = (q: string) => {
    navigate(`/?q=${encodeURIComponent(q)}`);
  };

  return (
    <div className="page-container" style={{ animation: 'fadeIn 0.2s ease-out' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <h1 className="page-title">Search History</h1>
        {history.length > 0 && (
          <button
            className="btn btn-secondary"
            style={{ fontSize: '12px', gap: '6px' }}
            onClick={clearHistory}
          >
            <Trash2 size={14} /> Clear History
          </button>
        )}
      </div>

      <p className="page-subtitle">
        Review your past semantic queries and quickly jump back to results.
      </p>

      {history.length === 0 ? (
        <div className="empty-state-container" style={{ minHeight: '40vh' }}>
          <div className="empty-state-icon-box">
            <History size={28} />
          </div>
          <h2 className="empty-state-title">No search history yet</h2>
          <p className="empty-state-subtitle">
            Searches you make will appear here for fast reference.
          </p>
          <button className="btn btn-primary" onClick={() => navigate('/')}>
            Go to Search
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', maxWidth: '720px' }}>
          {history.map((item) => (
            <div
              key={item.id}
              className="result-card"
              style={{
                padding: '16px 20px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
              onClick={() => handleRunQuery(item.query)}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div className="file-type-icon">
                  <Search size={18} color="var(--primary)" />
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '15px', color: 'var(--text-primary)' }}>
                    {item.query}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    {item.timestamp} · {item.count} results found
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary)' }}>
                <span style={{ fontSize: '13px', fontWeight: 500 }}>Search again</span>
                <ArrowRight size={16} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
