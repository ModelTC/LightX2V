import React from 'react';
import { LayoutGrid, Users } from 'lucide-react';
import { WorkflowState } from '../../../types';
import { PRESET_WORKFLOWS } from '../../../preset_workflow';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { WorkflowCard } from './WorkflowCard';
import { Header } from '../common/Header';
import type { LightX2VResultRef } from '../../hooks/useWorkflowExecution';

interface DashboardProps {
  lang: Language;
  myWorkflows: WorkflowState[];
  communityWorkflows: WorkflowState[];
  activeTab: 'MY' | 'COMMUNITY' | 'PRESET';
  onToggleLang: () => void;
  onCreateWorkflow: () => void;
  onAIGenerate: () => void;
  onOpenWorkflow: (workflow: WorkflowState) => void;
  onDeleteWorkflow: (id: string, e: React.MouseEvent) => void;
  onToggleWorkflowVisibility?: (workflowId: string, visibility: 'private' | 'public') => void;
  onToggleThumbsup?: (workflowId: string) => void;
  onSetActiveTab: (tab: 'MY' | 'COMMUNITY' | 'PRESET') => void;
  isLoading?: boolean;
  onRefresh?: () => void;
  /** 纯前端模式不显示社区栏目 */
  hideCommunityTab?: boolean;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef, context?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>;
}

export const Dashboard: React.FC<DashboardProps> = ({
  lang,
  myWorkflows,
  communityWorkflows,
  activeTab,
  onToggleLang,
  onCreateWorkflow,
  onAIGenerate,
  onOpenWorkflow,
  onDeleteWorkflow,
  onToggleWorkflowVisibility,
  onToggleThumbsup,
  onSetActiveTab,
  isLoading = false,
  onRefresh,
  hideCommunityTab = false,
  resolveLightX2VResultRef
}) => {
  const { t } = useTranslation(lang);
  const accentColor = '#90dce1';
  const getTabColor = (tab: 'MY' | 'COMMUNITY' | 'PRESET') => {
    if (tab === 'COMMUNITY') return '#fbbf24';
    if (tab === 'PRESET') return '#a78bfa';
    return accentColor;
  };

  return (
    <div className="w-full h-full overflow-hidden flex flex-col bg-slate-950 text-slate-200 selection:bg-[#90dce1]/30 font-sans">
      <Header
        lang={lang}
        onToggleLang={onToggleLang}
        onCreateWorkflow={onCreateWorkflow}
        onAIGenerate={onAIGenerate}
      />
      <main className="flex-1 p-12 w-full overflow-y-auto custom-scrollbar relative">
        <div className="w-full max-w-7xl mx-auto space-y-12 pb-24 relative z-10">
          <div className={`grid border-b border-slate-800/60 w-full max-w-4xl mx-auto ${hideCommunityTab ? 'grid-cols-2' : 'grid-cols-3'}`}>
            <button
              onClick={() => onSetActiveTab('MY')}
              className={`pb-4 text-sm font-black uppercase tracking-widest transition-all relative flex items-center justify-center ${
                activeTab === 'MY' ? '' : 'text-slate-500 hover:text-slate-300'
              }`}
              style={{ color: activeTab === 'MY' ? getTabColor('MY') : undefined }}
            >
              {t('my_workflows')}
              {activeTab === 'MY' && (
                <div className="absolute bottom-0 left-0 w-full h-1 rounded-full" style={{ backgroundColor: getTabColor('MY') }}></div>
              )}
            </button>
            {!hideCommunityTab && (
              <button
                onClick={() => onSetActiveTab('COMMUNITY')}
                className={`pb-4 text-sm font-black uppercase tracking-widest transition-all relative flex items-center justify-center ${
                  activeTab === 'COMMUNITY' ? '' : 'text-slate-500 hover:text-slate-300'
                }`}
                style={{ color: activeTab === 'COMMUNITY' ? getTabColor('COMMUNITY') : undefined }}
              >
                {t('workflow_community')}
                {activeTab === 'COMMUNITY' && (
                  <div className="absolute bottom-0 left-0 w-full h-1 rounded-full" style={{ backgroundColor: getTabColor('COMMUNITY') }}></div>
                )}
              </button>
            )}
            <button
              onClick={() => onSetActiveTab('PRESET')}
              className={`pb-4 text-sm font-black uppercase tracking-widest transition-all relative flex items-center justify-center ${
                activeTab === 'PRESET' ? '' : 'text-slate-500 hover:text-slate-300'
              }`}
              style={{ color: activeTab === 'PRESET' ? getTabColor('PRESET') : undefined }}
            >
              {t('preset_library')}
              {activeTab === 'PRESET' && (
                <div className="absolute bottom-0 left-0 w-full h-1 rounded-full" style={{ backgroundColor: getTabColor('PRESET') }}></div>
              )}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 w-full min-w-full">
            {activeTab === 'MY' ? (
              myWorkflows.length === 0 ? (
                <div className="col-span-full py-32 flex flex-col items-center justify-center opacity-20">
                  <LayoutGrid size={64} className="mb-4" />
                  <p className="text-sm font-black uppercase tracking-[0.3em]">{t('no_workflows')}</p>
                </div>
              ) : (
                [...myWorkflows]
                  .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0))
                  .map((w) => (
                  <WorkflowCard
                    key={w.id}
                    workflow={w}
                    lang={lang}
                    onOpen={onOpenWorkflow}
                    onDelete={onDeleteWorkflow}
                    onToggleVisibility={onToggleWorkflowVisibility}
                    onToggleThumbsup={onToggleThumbsup}
                    mode="MY"
                    accentColor={accentColor}
                    resolveLightX2VResultRef={resolveLightX2VResultRef}
                  />
                  ))
              )
            ) : !hideCommunityTab && activeTab === 'COMMUNITY' ? (
              communityWorkflows.length === 0 ? (
                <div className="col-span-full py-32 flex flex-col items-center justify-center opacity-20">
                  <Users size={64} className="mb-4" />
                  <p className="text-sm font-black uppercase tracking-[0.3em]">{t('no_community')}</p>
                </div>
              ) : (
                communityWorkflows.map((w) => (
                  <WorkflowCard
                    key={w.id}
                    workflow={w}
                    lang={lang}
                    onOpen={onOpenWorkflow}
                    onDelete={() => {}}
                    onToggleThumbsup={onToggleThumbsup}
                    mode="COMMUNITY"
                    accentColor={accentColor}
                    resolveLightX2VResultRef={resolveLightX2VResultRef}
                  />
                ))
              )
            ) : (
              PRESET_WORKFLOWS.map((w) => (
                <WorkflowCard
                  key={w.id}
                  workflow={w}
                  lang={lang}
                  onOpen={onOpenWorkflow}
                  onDelete={() => {}} // Presets cannot be deleted
                  isPreset={true}
                  mode="PRESET"
                  accentColor={accentColor}
                  resolveLightX2VResultRef={resolveLightX2VResultRef}
                />
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  );
};
