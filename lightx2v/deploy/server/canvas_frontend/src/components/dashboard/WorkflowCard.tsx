import React, { useState, useEffect } from 'react';
import { Layers, Trash2, Calendar, Sparkle, Lock, Globe, Heart, Users, Boxes } from 'lucide-react';
import { WorkflowState } from '../../../types';
import { useTranslation, Language } from '../../i18n/useTranslation';
import { getAssetPath } from '../../utils/assetPath';
import { getLocalFileDataUrl, getNodeOutputUrl } from '../../utils/workflowFileManager';
import type { LightX2VResultRef } from '../../hooks/useWorkflowExecution';

interface WorkflowCardProps {
  workflow: WorkflowState;
  lang: Language;
  onOpen: (workflow: WorkflowState) => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
  isPreset?: boolean;
  onToggleVisibility?: (workflowId: string, visibility: 'private' | 'public') => void;
  onToggleThumbsup?: (workflowId: string) => void;
  mode?: 'MY' | 'COMMUNITY' | 'PRESET';
  accentColor?: string;
  resolveLightX2VResultRef?: (ref: LightX2VResultRef, context?: { workflow_id?: string; node_id?: string; port_id?: string }) => Promise<string>;
}

export const WorkflowCard: React.FC<WorkflowCardProps> = ({
  workflow,
  lang,
  onOpen,
  onDelete,
  isPreset = false,
  onToggleVisibility,
  onToggleThumbsup,
  mode = 'MY',
  accentColor = '#90dce1',
  resolveLightX2VResultRef
}) => {
  const { t } = useTranslation(lang);

  // Extract preview content from workflow (image first, then text)
  const textInputNode = workflow.nodes.find(n => n.tool_id === 'text-input' && n.data?.value);
  const imageInputNode = workflow.nodes.find(n => n.tool_id === 'image-input' && (n.data?.value || (Array.isArray(n.data?.image_edits) && n.data.image_edits.length > 0)));
  const imageValue = imageInputNode?.data?.value;
  const imageEdits = Array.isArray(imageInputNode?.data?.image_edits) ? imageInputNode.data.image_edits : [];

  const previewText = typeof textInputNode?.data?.value === 'string' ? textInputNode?.data?.value : null;
  // 优先从 value 取第一张图：可为字符串(data:/local:/path)、file_ref(file_id) 或 task_ref(task_id+output_name)
  let previewImage: string | null = null;
  let previewFileRef: { file_id: string; mime_type?: string; ext?: string; run_id?: string; nodeId?: string; portId?: string } | null = null;
  let previewTaskRef: { task_id: string; output_name: string; nodeId?: string; portId?: string } | null = null;
  if (Array.isArray(imageValue) && imageValue.length > 0) {
    const first = imageValue[0];
    if (typeof first === 'string') previewImage = first;
    else if (first && typeof first === 'object' && typeof (first as any).file_id === 'string') {
      previewFileRef = {
        file_id: (first as any).file_id,
        mime_type: (first as any).mime_type,
        ext: (first as any).ext,
        run_id: (first as any).run_id,
        nodeId: imageInputNode?.id,
        portId: 'out-image',
      };
    } else if (first && typeof first === 'object' && typeof (first as any).task_id === 'string' && typeof (first as any).output_name === 'string') {
      previewTaskRef = { task_id: (first as any).task_id, output_name: (first as any).output_name, nodeId: imageInputNode?.id ?? undefined, portId: 'out-image' };
    }
  } else if (typeof imageValue === 'string') {
    previewImage = imageValue;
  } else if (imageValue && typeof imageValue === 'object' && typeof (imageValue as any).file_id === 'string') {
    previewFileRef = {
      file_id: (imageValue as any).file_id,
      mime_type: (imageValue as any).mime_type,
      ext: (imageValue as any).ext,
      run_id: (imageValue as any).run_id,
      nodeId: imageInputNode?.id,
      portId: 'out-image',
    };
  } else if (imageValue && typeof imageValue === 'object' && typeof (imageValue as any).task_id === 'string' && typeof (imageValue as any).output_name === 'string') {
    previewTaskRef = { task_id: (imageValue as any).task_id, output_name: (imageValue as any).output_name, nodeId: imageInputNode?.id ?? undefined, portId: 'out-image' };
  }
  if (!previewImage && !previewFileRef && !previewTaskRef && imageEdits.length > 0) {
    const first = imageEdits[0] as { cropped?: string; original?: string; source?: string };
    const s = first.cropped || first.original || first.source;
    previewImage = (typeof s === 'string' && !s.startsWith('data:')) ? s : null;
  }
  // 若 image-input 无图，则用第一个带图片输出的节点的 output_value（file_ref/task_ref）作为缩略图
  if (!previewImage && !previewFileRef && !previewTaskRef && workflow.id) {
    for (const node of workflow.nodes || []) {
      const ov = node.output_value;
      if (!ov || typeof ov !== 'object') continue;
      const imgVal = (ov as Record<string, unknown>)['out-image'];
      const ref = Array.isArray(imgVal) ? imgVal[0] : imgVal;
      if (ref && typeof ref === 'object') {
        if ((ref as any).kind === 'url' && typeof (ref as any).url === 'string') {
          previewImage = (ref as any).url;
          break;
        }
        if (typeof (ref as any).file_id === 'string') {
          previewFileRef = { file_id: (ref as any).file_id, mime_type: (ref as any).mime_type, ext: (ref as any).ext, run_id: (ref as any).run_id, nodeId: node.id, portId: 'out-image' };
          break;
        }
        if (typeof (ref as any).task_id === 'string' && typeof (ref as any).output_name === 'string') {
          previewTaskRef = { task_id: (ref as any).task_id, output_name: (ref as any).output_name, nodeId: node.id, portId: 'out-image' };
          break;
        }
      }
    }
  }

  // Resolve local:// (IndexedDB) to data URL for thumbnail
  const [resolvedLocalUrl, setResolvedLocalUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!previewImage || typeof previewImage !== 'string' || !previewImage.startsWith('local://')) {
      setResolvedLocalUrl(null);
      return;
    }
    let cancelled = false;
    getLocalFileDataUrl(previewImage).then((url) => {
      if (!cancelled && url) setResolvedLocalUrl(url);
      else if (!cancelled) setResolvedLocalUrl(null);
    });
    return () => { cancelled = true; };
  }, [previewImage]);

  // file_ref 预览：统一通过 /output/{port_id}/url 获取展示 URL
  const [resolvedFileRefUrl, setResolvedFileRefUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!previewFileRef?.nodeId || !previewFileRef?.portId || !workflow.id) {
      setResolvedFileRefUrl(null);
      return;
    }
    let cancelled = false;
    getNodeOutputUrl(workflow.id, previewFileRef.nodeId, previewFileRef.portId, previewFileRef.file_id, previewFileRef.run_id).then((url) => {
      if (!cancelled && url) setResolvedFileRefUrl(url);
      else if (!cancelled) setResolvedFileRefUrl(null);
    });
    return () => { cancelled = true; };
  }, [workflow.id, previewFileRef?.file_id, previewFileRef?.nodeId, previewFileRef?.portId, previewFileRef?.run_id]);

  // task_ref 预览：通过 resolveLightX2VResultRef 获取 URL
  const [resolvedTaskRefUrl, setResolvedTaskRefUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!previewTaskRef || !resolveLightX2VResultRef || !workflow.id || !previewTaskRef.nodeId || !previewTaskRef.portId) {
      setResolvedTaskRefUrl(null);
      return;
    }
    let cancelled = false;
    const ref: LightX2VResultRef = { kind: 'task', task_id: previewTaskRef.task_id, output_name: previewTaskRef.output_name, is_cloud: false };
    const ctx = { workflow_id: workflow.id, node_id: previewTaskRef.nodeId, port_id: previewTaskRef.portId };
    resolveLightX2VResultRef(ref, ctx).then((url) => {
      if (!cancelled && url) setResolvedTaskRefUrl(url);
      else if (!cancelled) setResolvedTaskRefUrl(null);
    }).catch(() => { if (!cancelled) setResolvedTaskRefUrl(null); });
    return () => { cancelled = true; setResolvedTaskRefUrl(null); };
  }, [workflow.id, previewTaskRef?.task_id, previewTaskRef?.output_name, previewTaskRef?.nodeId, previewTaskRef?.portId, resolveLightX2VResultRef]);

  // 显示用 URL：字符串(data:/local:// 解析后)、file_ref 用 getNodeOutputUrl 结果，task_ref 用 resolveLightX2VResultRef
  const displayImageUrl =
    (previewImage && typeof previewImage === 'string' && (previewImage.startsWith('local://') ? resolvedLocalUrl : previewImage))
    || (previewFileRef && resolvedFileRefUrl)
    || (previewTaskRef && resolvedTaskRefUrl) || null;

  const cardMode = isPreset ? 'PRESET' : mode;
  const visibility = workflow.visibility || (cardMode !== 'MY' ? 'public' : 'private');
  const isPublic = visibility === 'public';
  const nextVisibility = isPublic ? 'private' : 'public';
  const thumsupCount = workflow.thumsupCount ?? 0;
  const thumsupLiked = workflow.thumsupLiked ?? false;
  const authorName = workflow.authorName || workflow.authorId || 'Anonymous';
  const showThumbsup = visibility === 'public';
  const categoryColor = cardMode === 'PRESET'
    ? '#a78bfa'
    : cardMode === 'COMMUNITY'
    ? '#fbbf24'
    : accentColor;

  return (
    <div
      onClick={(e) => {
        const target = e.target as HTMLElement | null;
        if (target?.closest?.('[data-card-action="true"]')) return;
        onOpen(workflow);
      }}
      className="group flex flex-col bg-slate-900/50 border rounded-[32px] overflow-hidden cursor-pointer transition-all hover:shadow-[0_30px_60px_rgba(0,0,0,0.6)] active:scale-[0.98] aspect-[9/16] relative w-full"
      style={{ borderColor: `${categoryColor}22` }}
    >
      {cardMode === 'COMMUNITY' && (
        <div className="absolute inset-0 bg-gradient-to-br from-violet-600/5 to-transparent pointer-events-none" />
      )}

      <div className="relative flex-1 bg-slate-950/50 flex items-center justify-center overflow-hidden">
        {displayImageUrl ? (
          (() => {
            // data:image、local:// 解析结果、http(s)/blob 原样；file_ref/task_ref 为 /api/... 或 /assets/... 用 getAssetPath 加 token；路径用 getAssetPath；裸 base64 加 data: 前缀
            const raw =
              (typeof previewImage === 'string' && previewImage.startsWith('local://') ? displayImageUrl : null)
              ?? (typeof previewImage === 'string' ? previewImage : null)
              ?? (displayImageUrl || '');
            const src =
              typeof raw === 'string' && (raw.startsWith('data:') || raw.startsWith('http') || raw.startsWith('blob:'))
                ? raw
                : typeof raw === 'string' && (raw.startsWith('/') || raw.includes('/'))
                  ? getAssetPath(raw)
                  : typeof raw === 'string' && /^[A-Za-z0-9+/=]+$/.test(raw)
                    ? `data:image/png;base64,${raw}`
                    : getAssetPath(raw);
            return (
              <img
                src={src}
                alt="Preview"
                className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
              />
            );
          })()
        ) : previewText ? (
          <div className="px-10 py-8 text-[13px] text-slate-900 italic line-clamp-[10] leading-relaxed bg-white w-full h-full border-b border-slate-800/50 flex items-center justify-center text-center">
            {previewText}
          </div>
        ) : (
          <div className="p-8 text-[10px] text-slate-700 font-bold uppercase tracking-widest text-center opacity-30 flex flex-col items-center">
            <Boxes size={48} className="mb-4" /> No Preview Data
          </div>
        )}
        <div
          className="absolute top-5 left-5 p-2.5 backdrop-blur-2xl rounded-2xl shadow-2xl flex items-center justify-center transition-all group-hover:scale-110 z-10"
          style={{ backgroundColor: `${categoryColor}cc`, color: '#000' }}
        >
          {cardMode === 'PRESET'
            ? <Sparkle size={20} />
            : cardMode === 'COMMUNITY'
            ? <Users size={20} />
            : (isPublic ? <Globe size={20} /> : <Lock size={20} />)}
        </div>
      </div>

      <div className="p-7 space-y-4 shrink-0 bg-slate-900/80 backdrop-blur-md relative border-t border-white/5">
        <div>
          <div className="flex items-center justify-between mb-1 gap-3">
            <h3 className="text-lg font-black text-slate-100 group-hover:text-white transition-colors truncate flex-1 tracking-tight">
              {workflow.name}
            </h3>
            {cardMode === 'MY' && (
              <div className={`px-2.5 py-1 rounded-full text-[9px] font-black uppercase tracking-tighter ${
                isPublic ? 'bg-green-500/20 text-green-400' : 'bg-slate-800 text-slate-500'
              }`}>
                {isPublic ? t('visibility_public') : t('visibility_private')}
              </div>
            )}
          </div>

          {cardMode === 'COMMUNITY' ? (
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center overflow-hidden shadow-lg">
                  <img
                    src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${authorName}&backgroundColor=1e293b`}
                    alt="Author"
                    className="w-full h-full"
                  />
                </div>
                <span className="text-[11px] font-black text-slate-400 uppercase tracking-tight">
                  {authorName}
                </span>
              </div>
              <div className="flex items-center gap-3">
                {showThumbsup && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleThumbsup?.(workflow.id);
                    }}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded-xl transition-all duration-300 hover:bg-rose-500/10 active:scale-90 ${
                      thumsupLiked ? 'text-rose-500' : 'text-slate-500 hover:text-rose-400'
                    }`}
                    title="Thumsup"
                  >
                    <Heart size={16} className={thumsupLiked ? 'fill-rose-500' : ''} />
                    <span className="text-[11px] font-black">{thumsupCount}</span>
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2 text-[11px] font-bold text-slate-500 uppercase tracking-widest">
                <Calendar size={12} /> {new Date(workflow.updatedAt).toLocaleDateString()}
              </div>
              <div className="flex items-center gap-2">
                {showThumbsup && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleThumbsup?.(workflow.id);
                    }}
                    className={`flex items-center gap-1.5 px-2 py-1.5 rounded-xl transition-all duration-300 hover:bg-rose-500/10 active:scale-90 ${
                      thumsupLiked ? 'text-rose-500' : 'text-slate-500 hover:text-rose-400'
                    }`}
                    title="Thumsup"
                  >
                    <Heart size={16} className={thumsupLiked ? 'fill-rose-500' : ''} />
                    <span className="text-[11px] font-black">{thumsupCount}</span>
                  </button>
                )}
                {cardMode === 'MY' && (
                  <button
                    data-card-action="true"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(workflow.id, e);
                    }}
                    className="p-2.5 text-slate-700 hover:text-red-400 transition-all hover:bg-red-400/5 rounded-xl active:scale-90"
                  >
                    <Trash2 size={18} />
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
