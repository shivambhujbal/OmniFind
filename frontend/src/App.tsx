import { useState } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Menu } from 'lucide-react';
import Sidebar from './components/Sidebar';
import SettingsModal from './components/SettingsModal';
import AboutModal from './components/AboutModal';
import SearchPage from './pages/SearchPage';
import SearchHistoryPage from './pages/SearchHistoryPage';
import LibraryPage from './pages/LibraryPage';
import StatusPage from './pages/StatusPage';

function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const location = useLocation();

  const isHome = location.pathname === '/' && !location.search;

  return (
    <>
      {/* Top Ambient Glow */}
      <div className="ambient-glow" />

      {/* Floating Minimalist Header (Takes no layout space) */}
      <header className="top-header">
        {!isHome ? (
          <Link to="/" className="top-brand" title="OmniFind Home">
            <span className="top-brand-title">OmniFind</span>
          </Link>
        ) : (
          <div /> /* Empty placeholder so hamburger stays on right */
        )}

        <button
          className="menu-toggle-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          title={sidebarOpen ? 'Close Navigation' : 'Open Navigation'}
          aria-label="Toggle navigation menu"
        >
          <Menu size={22} strokeWidth={1.8} />
        </button>
      </header>

      <div className="app-layout">
        {/* Main Content Area */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/history" element={<SearchHistoryPage />} />
            <Route path="/library" element={<LibraryPage />} />
            <Route path="/status" element={<StatusPage />} />
          </Routes>
        </main>

        {/* Right Navigation Sidebar */}
        {sidebarOpen && (
          <Sidebar
            onOpenSettings={() => setSettingsOpen(true)}
            onOpenAbout={() => setAboutOpen(true)}
          />
        )}
      </div>

      {/* Settings & About Modals */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      <AboutModal
        isOpen={aboutOpen}
        onClose={() => setAboutOpen(false)}
      />
    </>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
