import { NavLink } from 'react-router-dom';
import {
  Home,
  History,
  Folder,
  RotateCw,
  Settings,
  Info,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { getHealth, BackendError } from '../api/client';
import type { HealthResponse } from '../api/types';

interface SidebarProps {
  onOpenSettings: () => void;
  onOpenAbout: () => void;
}

export default function Sidebar({ onOpenSettings, onOpenAbout }: SidebarProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const h = await getHealth();
        if (!cancelled) {
          setHealth(h);
          setConnected(true);
        }
      } catch (e) {
        if (!cancelled) {
          setHealth(null);
          setConnected(false);
        }
      }
    };

    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-brand">OmniFind</div>
        <div className="sidebar-subbrand">Local Search Utility</div>
      </div>

      <nav className="sidebar-nav">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `sidebar-item${isActive ? ' active' : ''}`}
        >
          <Home size={18} />
          <span>Home</span>
        </NavLink>

        <NavLink
          to="/history"
          className={({ isActive }) => `sidebar-item${isActive ? ' active' : ''}`}
        >
          <History size={18} />
          <span>Search History</span>
        </NavLink>

        <NavLink
          to="/library"
          className={({ isActive }) => `sidebar-item${isActive ? ' active' : ''}`}
        >
          <Folder size={18} />
          <span>Indexed Folders</span>
        </NavLink>

        <NavLink
          to="/status"
          className={({ isActive }) => `sidebar-item${isActive ? ' active' : ''}`}
        >
          <RotateCw size={18} />
          <span>Indexing Status</span>
        </NavLink>

        <button
          type="button"
          className="sidebar-item"
          onClick={onOpenSettings}
        >
          <Settings size={18} />
          <span>Settings</span>
        </button>

        <button
          type="button"
          className="sidebar-item"
          onClick={onOpenAbout}
        >
          <Info size={18} />
          <span>About</span>
        </button>
      </nav>

      <div className="sidebar-footer">
        <div className="connection-pill">
          <span
            className={`status-dot ${
              connected === true ? 'connected' : connected === false ? 'disconnected' : ''
            }`}
          />
          <span>
            {connected === true
              ? 'Local Sidecar Connected'
              : connected === false
              ? 'Sidecar Disconnected'
              : 'Connecting...'}
          </span>
        </div>
      </div>
    </aside>
  );
}
