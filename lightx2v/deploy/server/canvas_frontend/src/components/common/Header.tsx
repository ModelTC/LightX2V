import React, { useState, useEffect } from 'react';
import { Wand2, Languages, Sparkle, Plus, LayoutGrid, Github, Gift, CreditCard } from 'lucide-react';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { isStandalone } from '../../config/runtimeMode';
import { UserCard } from './UserCard';

interface HeaderProps {
  lang: Language;
  onToggleLang: () => void;
  onCreateWorkflow: () => void;
  onAIGenerate: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  lang,
  onToggleLang,
  onCreateWorkflow,
  onAIGenerate
}) => {
  const { t } = useTranslation(lang);
  const [user, setUser] = useState<any>(null);

  // 获取用户信息
  useEffect(() => {
    const sharedStore = (window as any).__SHARED_STORE__;

    // 初始化用户信息
    if (sharedStore) {
      const currentUser = sharedStore.getState('user');
      setUser(currentUser);

      // 订阅用户状态变化
      const unsubscribe = sharedStore.subscribe((state: any) => {
        setUser(state.user);
      });

      return () => {
        if (unsubscribe) unsubscribe();
      };
    } else {
      // 如果没有 sharedStore，从 localStorage 读取
      try {
        const userStr = localStorage.getItem('currentUser');
        if (userStr) {
          setUser(JSON.parse(userStr));
        }
      } catch (e) {
        console.error('Failed to parse user from localStorage:', e);
      }
    }
  }, []);

  // 切换到普通模式（主应用的 generate 页面）
  const handleSwitchToNormalMode = () => {
    // 在 qiankun 环境中，使用 window.location 跳转到主应用
    // 如果在 iframe 中，尝试使用 parent，否则使用当前窗口
    try {
      if (window.parent !== window) {
        // 在 iframe 中
        window.parent.location.href = '/generate';
      } else {
        // 直接跳转
        window.location.href = '/generate';
      }
    } catch (e) {
      // 跨域限制，使用当前窗口
      window.location.href = '/generate';
    }
  };

  // 处理登录
  const handleLogin = () => {
    // 跳转到主应用的登录页面，然后返回当前页面
    try {
      if (window.parent !== window) {
        window.parent.location.href = '/generate';
      } else {
        window.location.href = '/generate';
      }
    } catch (e) {
      window.location.href = '/generate';
    }
  };

  // 处理登出
  const handleLogout = () => {
    const sharedStore = (window as any).__SHARED_STORE__;
    if (sharedStore) {
      sharedStore.clear();
    } else {
      localStorage.removeItem('accessToken');
      localStorage.removeItem('currentUser');
    }
    setUser(null);
    // 刷新页面或跳转到登录页
    window.location.reload();
  };

  return (
    <header className="h-16 border-b border-slate-800/80 flex items-center justify-between px-8 bg-slate-950/70 backdrop-blur-3xl z-40">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-[#90dce1] rounded-2xl flex items-center justify-center shadow-xl shadow-[#90dce1]/20 transition-all hover:shadow-[#90dce1]/40 hover:-translate-y-0.5">
          <Wand2 className="text-slate-900" size={20} />
        </div>
        <div className="flex flex-col leading-none">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-black uppercase tracking-widest text-white">
              {t('app_name')}
            </h1>
            <span className="text-[9px] font-black uppercase tracking-[0.2em] px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[#90dce1] transition-all hover:border-[#90dce1]/60 hover:shadow-[0_0_14px_rgba(144,220,225,0.35)]">
              PRO
            </span>
          </div>
          <span className="text-[9px] text-slate-500 font-bold uppercase tracking-[0.2em]">
            {t('app_subtitle')}
          </span>
        </div>
        {!isStandalone() && (
          <div className="hidden md:flex items-center bg-slate-900/60 border border-slate-800/80 rounded-2xl p-1 ml-2">
            <button
              onClick={handleSwitchToNormalMode}
              className="px-3 py-1.5 text-[9px] font-black uppercase tracking-widest text-slate-500 flex items-center gap-2 hover:text-slate-300 transition-all hover:bg-slate-900/80 rounded-xl"
              title={lang === 'zh' ? '切换到普通模式' : 'Switch to Normal Mode'}
            >
              <Gift size={12} className="text-slate-600" />
              {t('lite_free_label')}
            </button>
            <div className="px-3 py-1.5 text-[9px] font-black uppercase tracking-widest text-slate-900 bg-[#90dce1] rounded-xl shadow-[0_0_20px_rgba(144,220,225,0.25)]">
              <CreditCard size={12} className="text-slate-900 inline-block mr-2" />
              {t('pro_edition_label')}
            </div>
          </div>
        )}
      </div>
      <div className="flex items-center gap-3">
        <a
          href="https://github.com/ModelTC/LightX2V"
          target="_blank"
          className="flex items-center gap-2 px-4 py-2 bg-slate-900/60 text-slate-300 rounded-2xl text-[11px] font-bold uppercase tracking-widest transition-all border border-slate-800/80 group hover:border-[#90dce1]/60 hover:shadow-[0_0_18px_rgba(144,220,225,0.2)] hover:-translate-y-0.5"
        >
          <Github size={14} className="group-hover:text-[#90dce1]" />
          <span className="hidden lg:inline">{t('visit_github')}</span>
        </a>
        <button
          onClick={onToggleLang}
          className="flex items-center gap-2 px-4 py-2 bg-slate-900/60 text-slate-300 rounded-2xl text-[11px] font-bold uppercase tracking-widest transition-all border border-slate-800/80 hover:border-[#90dce1]/60 hover:shadow-[0_0_18px_rgba(144,220,225,0.2)] hover:-translate-y-0.5"
        >
          <Languages size={14} /> {t('lang_name')}
        </button>
        <button
          onClick={onAIGenerate}
          className="flex items-center gap-2 px-5 py-2 bg-slate-900/70 text-slate-200 rounded-2xl text-[11px] font-bold uppercase tracking-widest transition-all border-2 border-[#90dce1]/80 shadow-[0_8px_20px_rgba(15,23,42,0.35)] hover:shadow-[0_0_24px_rgba(144,220,225,0.35)] hover:-translate-y-0.5"
        >
          <Sparkle size={18} /> {t('ai_generate_workflow')}
        </button>
        <button
          onClick={onCreateWorkflow}
          className="flex items-center gap-2 px-7 py-2.5 bg-[#90dce1] text-slate-900 rounded-2xl text-[11px] font-black uppercase tracking-widest transition-all active:scale-95 shadow-[0_10px_30px_rgba(144,220,225,0.35)] hover:bg-[#7dd3da] hover:shadow-[0_0_28px_rgba(144,220,225,0.5)] hover:-translate-y-0.5"
        >
          <Plus size={18} /> {t('create_workflow')}
        </button>

        {!isStandalone() && (
          <UserCard
            user={user}
            lang={lang}
            onLogin={handleLogin}
            onLogout={handleLogout}
          />
        )}
      </div>
    </header>
  );
};
