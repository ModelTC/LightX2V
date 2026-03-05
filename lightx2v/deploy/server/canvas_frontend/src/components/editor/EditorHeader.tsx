import React, { useState, useEffect } from 'react';
import {
  ChevronLeft,
  ZoomIn,
  ZoomOut,
  Maximize,
  Languages,
  X,
  Save,
  Play,
  Pause,
  RefreshCw,
  Zap,
  Undo,
  Redo,
  LayoutGrid,
  Github,
  Hash,
  Square,
  Lock,
  Globe,
  Terminal
} from 'lucide-react';
import { WorkflowState } from '../../../types';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { isStandalone } from '../../config/runtimeMode';
import { UserCard } from '../common/UserCard';

interface ViewState {
  x: number;
  y: number;
  zoom: number;
}

interface EditorHeaderProps {
  lang: Language;
  workflow: WorkflowState;
  view: ViewState;
  isPaused: boolean;
  isRunning: boolean;
  isSaving?: boolean;
  canvasRef: React.RefObject<HTMLDivElement>;
  onBack: () => void;
  onWorkflowNameChange: (name: string) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetView: () => void;
  onToggleLang: () => void;
  onClearSnapshot: () => void;
  onSave: () => void;
  onPause: () => void;
  onRun: () => void;
  onStop: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onVisibilityChange: (isPublic: boolean) => void;
}

export const EditorHeader: React.FC<EditorHeaderProps> = ({
  lang,
  workflow,
  view,
  isPaused,
  isRunning,
  isSaving = false,
  canvasRef,
  onBack,
  onWorkflowNameChange,
  onZoomIn,
  onZoomOut,
  onResetView,
  onToggleLang,
  onClearSnapshot,
  onSave,
  onPause,
  onRun,
  onStop,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onVisibilityChange
}) => {
  const { t } = useTranslation(lang);
  const [user, setUser] = useState<any>(null);
  const isPublic = workflow.visibility === 'public';

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
    // 跳转到主应用的登录页面
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
    // 刷新页面
    window.location.reload();
  };

  return (
    <header className="h-16 border-b border-slate-800/80 flex items-center justify-between px-6 bg-slate-950/70 backdrop-blur-3xl z-40">
      <div className="flex items-center gap-5">
        <button
          onClick={onBack}
          className="p-2 text-slate-500 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
          title={lang === 'zh' ? '返回工作流列表' : 'Back to Workflows'}
        >
          <ChevronLeft size={20} />
        </button>
        <div className="flex flex-col">
          <input
            value={workflow.name}
            onChange={(e) => {
              onClearSnapshot();
              onWorkflowNameChange(e.target.value);
            }}
            className="bg-transparent border-none text-base font-bold focus:ring-0 p-0 hover:bg-slate-800/20 rounded px-1 transition-colors w-64"
          />
          <div className="flex items-center gap-2">
            <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">
              {t('editing_logic')}
            </span>
            {workflow.id && (workflow.id.startsWith('workflow-') || workflow.id.match(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i)) && (
              <span className="text-[9px] text-slate-600 font-mono">
                <Hash size={8} className="inline mr-0.5" />
                {workflow.id.substring(0, 8)}...
              </span>
            )}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <a
          href="https://github.com/ModelTC/LightX2V"
          target="_blank"
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 hover:bg-slate-900 text-slate-300 rounded-2xl text-[10px] font-bold uppercase tracking-widest transition-all border border-slate-800/80 group"
        >
          <Github size={12} className="group-hover:text-[#90dce1]" />
          <span className="hidden lg:inline">{t('visit_github')}</span>
        </a>
        {/* Undo/Redo Controls */}
        <div className="flex items-center gap-1 bg-slate-800/50 border border-slate-800 rounded-xl p-1">
          <button
            onClick={onUndo}
            disabled={!canUndo}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            title={lang === 'zh' ? '撤销 (Ctrl+Z)' : 'Undo (Ctrl+Z)'}
          >
            <Undo size={14} />
          </button>
          <button
            onClick={onRedo}
            disabled={!canRedo}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            title={lang === 'zh' ? '重做 (Ctrl+Y)' : 'Redo (Ctrl+Y)'}
          >
            <Redo size={14} />
          </button>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-1 bg-slate-800/50 border border-slate-800 rounded-xl p-1">
          <button
            onClick={onZoomIn}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '放大' : 'Zoom In'}
          >
            <ZoomIn size={14} />
          </button>
          <div className="px-2 py-1 text-[10px] font-bold text-slate-500 min-w-[3rem] text-center">
            {Math.round(view.zoom * 100)}%
          </div>
          <button
            onClick={onZoomOut}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '缩小' : 'Zoom Out'}
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={onResetView}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-all"
            title={lang === 'zh' ? '重置视图' : 'Reset View'}
          >
            <Maximize size={14} />
          </button>
        </div>

        <button
          onClick={onToggleLang}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 hover:bg-slate-900 text-slate-300 rounded-2xl text-[10px] font-bold uppercase tracking-widest transition-all border border-slate-800/80"
        >
          <Languages size={12} /> {t('lang_name')}
        </button>
        <button
          onClick={() => {
            console.log('[工作流] 当前工作流数据:', workflow);
          }}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/60 hover:bg-slate-900 text-slate-400 hover:text-slate-300 rounded-2xl text-[10px] font-bold uppercase tracking-widest transition-all border border-slate-800/80"
          title={lang === 'zh' ? '在控制台打印当前工作流数据' : 'Print current workflow to console'}
        >
          <Terminal size={12} />
          {lang === 'zh' ? '打印工作流' : 'Print'}
        </button>
        {!isStandalone() && (
          <div className="flex items-center bg-slate-950/50 p-1 rounded-2xl border border-slate-800/50">
            <button
              onClick={() => onVisibilityChange(false)}
              className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all ${
                !isPublic ? 'bg-slate-800 text-slate-200 shadow-xl' : 'text-slate-600 hover:text-slate-400'
              }`}
            >
              <Lock size={12} />
              <span className="text-[10px] font-black uppercase tracking-widest">{t('visibility_private')}</span>
            </button>
            <button
              onClick={() => onVisibilityChange(true)}
              className={`px-3 py-1.5 rounded-xl flex items-center gap-2 transition-all ${
                isPublic
                  ? 'bg-green-500/10 text-green-400 border border-green-500/20 shadow-xl shadow-green-500/5'
                  : 'text-slate-600 hover:text-slate-400'
              }`}
            >
              <Globe size={12} />
              <span className="text-[10px] font-black uppercase tracking-widest">{t('visibility_public')}</span>
            </button>
          </div>
        )}
        <button
          onClick={onSave}
          disabled={isSaving}
          className={`flex items-center gap-2 px-6 py-2 rounded-2xl text-[11px] font-bold uppercase tracking-widest border-2 transition-all shadow-[0_8px_20px_rgba(15,23,42,0.35)] ${
            isSaving
              ? 'bg-slate-800 border-slate-700 text-slate-500 cursor-not-allowed'
              : workflow.isDirty
              ? 'bg-slate-900/70 border-[#90dce1]/80 text-slate-200 hover:shadow-[0_0_20px_rgba(144,220,225,0.25)] hover:-translate-y-0.5'
              : 'bg-slate-900/70 border-slate-700/80 text-slate-500'
          }`}
        >
          <Save size={16} /> {t('save_flow')}
        </button>
        <div className="w-px h-6 bg-slate-800"></div>
        {isRunning && !isPaused && (
          <button
            onClick={onStop}
            className="flex items-center gap-2 px-6 py-2.5 rounded-2xl text-sm font-bold shadow-xl transition-all bg-red-600 hover:bg-red-500 text-white shadow-red-500/20 active:scale-95"
          >
            <Square size={16} />
            {lang === 'zh' ? '停止' : 'Stop'}
          </button>
        )}
        {isRunning && isPaused && (
          <button
            onClick={onPause}
            className="flex items-center gap-2 px-6 py-2.5 rounded-2xl text-sm font-bold shadow-xl transition-all bg-yellow-600 hover:bg-yellow-500 text-white shadow-yellow-500/20 active:scale-95"
          >
            <Play size={16} />
            {lang === 'zh' ? '继续' : 'Resume'}
          </button>
        )}
        <button
          onClick={onRun}
          disabled={isRunning && !isPaused}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-2xl text-[11px] font-black uppercase tracking-widest shadow-[0_10px_30px_rgba(144,220,225,0.35)] transition-all ${
            isRunning && !isPaused
              ? 'bg-slate-800 text-slate-500 shadow-none'
              : 'bg-[#90dce1] hover:bg-[#7dd3da] text-slate-900 hover:shadow-[0_0_28px_rgba(144,220,225,0.5)] hover:-translate-y-0.5 active:scale-95'
          }`}
        >
          {isRunning && !isPaused ? (
            <RefreshCw className="animate-spin" size={16} />
          ) : (
            <Zap size={16} />
          )}
          {isRunning && !isPaused ? t('executing') : t('run_fabric')}
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
