import { useState } from 'react';
import { Outlet, useLocation, Link } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { ToastContainer } from './Toast';
import { ErrorBoundary } from './ErrorBoundary';
import { NotificationCenter } from './NotificationCenter';
import { useStore } from '../store/useStore';
import { ChevronRight, WifiOff, Menu } from 'lucide-react';

function Breadcrumbs() {
  const location = useLocation();
  const parts = location.pathname.split('/').filter(Boolean);

  if (parts.length === 0) return null;

  const crumbs: { label: string; path: string }[] = [];
  let path = '';
  for (const p of parts) {
    path += `/${p}`;
    crumbs.push({ label: p.replace(/-/g, ' '), path });
  }

  return (
    <nav className="flex items-center gap-1 text-xs text-gray-500 mb-4">
      <Link to="/" className="hover:text-gray-300 transition-colors">Home</Link>
      {crumbs.map((c, i) => (
        <span key={c.path} className="flex items-center gap-1">
          <ChevronRight className="w-3 h-3" />
          {i === crumbs.length - 1 ? (
            <span className="text-gray-300 capitalize">{c.label}</span>
          ) : (
            <Link to={c.path} className="hover:text-gray-300 transition-colors capitalize">
              {c.label}
            </Link>
          )}
        </span>
      ))}
    </nav>
  );
}

export function Layout() {
  const connected = useStore((s) => s.connected);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — always visible on desktop, slide-over on mobile */}
      <div
        className={[
          'fixed inset-y-0 left-0 z-30 transition-transform duration-200 lg:static lg:translate-x-0',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full',
        ].join(' ')}
      >
        <Sidebar onClose={() => setSidebarOpen(false)} />
      </div>

      <main className="flex-1 p-4 lg:p-6 overflow-auto min-w-0">
        {/* Mobile top bar */}
        <div className="flex items-center gap-3 mb-4 lg:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors"
            aria-label="Open menu"
          >
            <Menu className="w-5 h-5" />
          </button>
          <span className="font-semibold text-sm text-gray-200 flex-1">GA Dashboard</span>
          <NotificationCenter />
        </div>

        {/* Desktop top bar — notification bell right-aligned */}
        <div className="hidden lg:flex items-center justify-end mb-2 -mt-2">
          <NotificationCenter />
        </div>

        {/* Reconnection banner */}
        {!connected && (
          <div className="mb-4 flex items-center gap-2 px-4 py-2.5 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-xs">
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            <span>Connection lost — reconnecting automatically...</span>
            <div className="ml-auto w-3 h-3 border-2 border-yellow-400/60 border-t-yellow-400 rounded-full animate-spin" />
          </div>
        )}
        <Breadcrumbs />
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>
      <ToastContainer />
    </div>
  );
}
