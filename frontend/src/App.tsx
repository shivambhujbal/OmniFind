import { useState } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Menu } from 'lucide-react';
import Sidebar from './components/Sidebar';
import SettingsModal from './components/SettingsModal';
import AboutModal from './components/AboutModal';
import SearchPage from './pages/SearchPage';
import SearchHistoryPage from './pages/SearchHistoryPage';
import LibraryPage from './pages/LibraryPage';
import StatusPage from './pages/StatusPage';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <BrowserRouter>
      {/* Top Ambient Glow from screenshots */}
      <div className="ambient-glow" />

      {/* Top Header Bar for brand & hamburger toggle */}
      <header className="top-header">
        <Link to="/" className="top-brand">
          <span className="top-brand-title">OmniFind</span>
        </Link>
        <button
          className="menu-toggle-btn"
          onClick={() => setSidebarOpen(!sidebarOpen)}
          title="Toggle Navigation Menu"
          aria-label="Toggle navigation menu"
        >
          <Menu size={20} />
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

        {/* Right Navigation Sidebar (Screenshots 1 & 3) */}
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
    </BrowserRouter>
  );
}
