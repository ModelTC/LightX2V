import React, { useEffect, useRef, useState } from 'react';
import { User, Settings as SettingsIcon, LogOut, LogIn, ShieldCheck } from 'lucide-react';
import { Language } from '../../i18n/useTranslation';

interface UserCardProps {
  user: any;
  lang: Language;
  onLogin: () => void;
  onLogout: () => void;
  accentColor?: string;
}

export const UserCard: React.FC<UserCardProps> = ({
  user,
  lang,
  onLogin,
  onLogout,
  accentColor = '#90dce1'
}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const displayName = user?.username || user?.email || (lang === 'zh' ? '未登录' : 'Not logged in');
  const initials = (user?.username?.[0] || user?.email?.[0] || '?').toUpperCase();
  const isLoggedIn = !!(user?.username || user?.email);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    if (isMenuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMenuOpen]);

  return (
    <div ref={menuRef} className="flex items-center gap-3 px-3 py-1.5 bg-slate-900/60 hover:bg-slate-900 border border-slate-800/80 rounded-2xl transition-all duration-200">
      <div className="flex flex-col items-end leading-none min-w-[140px]">
        <span className="text-[11px] font-black uppercase tracking-widest text-slate-200 whitespace-nowrap overflow-hidden text-ellipsis max-w-[140px]">
          {displayName}
        </span>
        <div className="flex items-center gap-1.5 mt-1 text-[9px] font-black uppercase tracking-[0.2em]" style={{ color: `${accentColor}` }}>
          <ShieldCheck size={10} style={{ color: `${accentColor}` }} />
          {lang === 'zh' ? 'PRO MEMBER' : 'PRO MEMBER'}
        </div>
      </div>
      <div className="relative">
        <div
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="w-9 h-9 rounded-full border-2 bg-slate-800 flex items-center justify-center overflow-hidden transition-all cursor-pointer shadow-[0_0_12px_rgba(144,220,225,0.25)]"
          style={{ borderColor: isMenuOpen ? accentColor : `${accentColor}44` }}
        >
          {user?.avatar_url ? (
            <img
              src={user.avatar_url}
              alt={displayName}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-600/50 to-slate-700/50">
              <span className="text-xs text-slate-300 font-medium">{initials}</span>
            </div>
          )}
        </div>
        <div
          className={`absolute top-full right-0 mt-3 w-52 bg-slate-900/95 backdrop-blur-xl border border-slate-800 rounded-2xl shadow-2xl transition-all duration-300 z-[100] overflow-hidden ring-1 ring-white/5 ${
            isMenuOpen ? 'opacity-100 translate-y-0 pointer-events-auto' : 'opacity-0 translate-y-2 pointer-events-none'
          }`}
        >
          <div className="p-2.5 space-y-1">
            <button
              onClick={() => setIsMenuOpen(false)}
              className="w-full flex items-center gap-3 p-3 text-xs font-bold text-slate-400 hover:text-white hover:bg-white/5 rounded-xl transition-all"
            >
              <User size={14} /> {lang === 'en' ? 'Profile' : '个人资料'}
            </button>
            <button
              onClick={() => setIsMenuOpen(false)}
              className="w-full flex items-center gap-3 p-3 text-xs font-bold text-slate-400 hover:text-white hover:bg-white/5 rounded-xl transition-all"
            >
              <SettingsIcon size={14} /> {lang === 'en' ? 'Account Settings' : '账号设置'}
            </button>
            <div className="h-px bg-slate-800 mx-2 my-1" />
            {isLoggedIn ? (
              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  onLogout();
                }}
                className="w-full flex items-center gap-3 p-3 text-xs font-bold text-red-400 hover:bg-red-400/10 rounded-xl transition-all"
              >
                <LogOut size={14} /> {lang === 'en' ? 'Logout' : '登出'}
              </button>
            ) : (
              <button
                onClick={() => {
                  setIsMenuOpen(false);
                  onLogin();
                }}
                className="w-full flex items-center gap-3 p-3 text-xs font-bold text-slate-400 hover:text-white hover:bg-white/5 rounded-xl transition-all"
              >
                <LogIn size={14} /> {lang === 'en' ? 'Login' : '登录'}
              </button>
            )}
          </div>
        </div>
      </div>
      <span className="sr-only" style={{ color: accentColor }}>
        {accentColor}
      </span>
    </div>
  );
};
