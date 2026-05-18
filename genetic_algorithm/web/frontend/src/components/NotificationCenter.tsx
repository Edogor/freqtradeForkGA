/**
 * NotificationCenter — bell icon in the Layout header that opens a dropdown
 * with persistent notification history (run completions, new bests,
 * overfitting warnings, HoF updates, errors).
 *
 * Notifications are driven by the global Zustand store which already
 * subscribes to the WebSocket event stream.
 */

import { useRef, useState, useEffect } from 'react';
import { Bell, X, CheckCheck, Trash2, CheckCircle2, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { useStore } from '../store/useStore';
import type { Notification } from '../store/useStore';

const iconMap = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
  warning: AlertTriangle,
};

const dotMap = {
  success: 'bg-green-400',
  error: 'bg-red-400',
  info: 'bg-blue-400',
  warning: 'bg-yellow-400',
};

const iconColorMap = {
  success: 'text-green-400',
  error: 'text-red-400',
  info: 'text-blue-400',
  warning: 'text-yellow-400',
};

function timeAgo(ms: number): string {
  const secs = Math.floor((Date.now() - ms) / 1000);
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function NotificationRow({ n }: { n: Notification }) {
  const Icon = iconMap[n.type];
  return (
    <div
      className={`flex items-start gap-2.5 px-3 py-2.5 border-b border-white/5 last:border-0 transition-colors ${
        n.read ? 'opacity-60' : 'bg-white/[0.02]'
      }`}
    >
      {!n.read && (
        <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotMap[n.type]}`} />
      )}
      {n.read && <span className="w-1.5 h-1.5 flex-shrink-0" />}
      <Icon className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${iconColorMap[n.type]}`} />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-gray-200 leading-snug">{n.title}</p>
        {n.message && (
          <p className="text-[11px] text-gray-400 mt-0.5 truncate">{n.message}</p>
        )}
      </div>
      <span className="text-[10px] text-gray-600 flex-shrink-0 mt-0.5">{timeAgo(n.timestamp)}</span>
    </div>
  );
}

export function NotificationCenter() {
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const notifications = useStore((s) => s.notifications);
  const unreadCount = useStore((s) => s.unreadCount);
  const markAllRead = useStore((s) => s.markAllRead);
  const clearNotifications = useStore((s) => s.clearNotifications);

  // Close when clicking outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        panelRef.current && !panelRef.current.contains(e.target as Node) &&
        buttonRef.current && !buttonRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Mark all read when panel opens
  useEffect(() => {
    if (open && unreadCount > 0) {
      // Slight delay so the badge is still visible when user opens
      const t = setTimeout(markAllRead, 800);
      return () => clearTimeout(t);
    }
  }, [open]);

  return (
    <div className="relative">
      {/* Bell button */}
      <button
        ref={buttonRef}
        onClick={() => setOpen((v) => !v)}
        className="relative p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors"
        aria-label="Notifications"
      >
        <Bell className="w-4 h-4" />
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 flex items-center justify-center rounded-full bg-accent text-[9px] font-bold text-white leading-none">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown panel */}
      {open && (
        <div
          ref={panelRef}
          className="absolute right-0 top-full mt-2 w-80 max-h-[480px] flex flex-col z-50
            bg-surface-1 border border-white/10 rounded-xl shadow-2xl overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/10 flex-shrink-0">
            <span className="text-xs font-semibold text-gray-300">Notifications</span>
            <div className="flex items-center gap-1">
              {notifications.length > 0 && (
                <>
                  <button
                    onClick={markAllRead}
                    title="Mark all read"
                    className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    <CheckCheck className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={clearNotifications}
                    title="Clear all"
                    className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
              <button
                onClick={() => setOpen(false)}
                className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Body */}
          <div className="overflow-y-auto flex-1">
            {notifications.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-10 gap-2 text-gray-600">
                <Bell className="w-6 h-6 opacity-30" />
                <span className="text-xs">No notifications yet</span>
              </div>
            ) : (
              notifications.map((n) => <NotificationRow key={n.id} n={n} />)
            )}
          </div>
        </div>
      )}
    </div>
  );
}
