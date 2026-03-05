import { useState } from 'react';

export const useModalState = () => {
  const [showCloneVoiceModal, setShowCloneVoiceModal] = useState(false);
  const [showAudioEditor, setShowAudioEditor] = useState<string | null>(null); // nodeId of audio input being edited
  const [showVideoEditor, setShowVideoEditor] = useState<string | null>(null); // nodeId of video input being edited
  const [expandedOutput, setExpandedOutput] = useState<{ nodeId: string; fieldId?: string; historyEntryId?: string } | null>(null);
  const [isEditingResult, setIsEditingResult] = useState(false);
  const [tempEditValue, setTempEditValue] = useState("");
  const [showReplaceMenu, setShowReplaceMenu] = useState<string | null>(null);
  const [showOutputQuickAdd, setShowOutputQuickAdd] = useState<{ nodeId: string; portId: string } | null>(null);
  const [showModelSelect, setShowModelSelect] = useState<string | null>(null);
  const [showVoiceSelect, setShowVoiceSelect] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [resultsCollapsed, setResultsCollapsed] = useState(true);
  const [isAIChatOpen, setIsAIChatOpen] = useState(false);
  const [isAIChatCollapsed, setIsAIChatCollapsed] = useState(false);
  const [nodeConfigPanelCollapsed, setNodeConfigPanelCollapsed] = useState(false);

  // AI Chat Panel 位置和大小（浮动窗口）
  const [aiChatPanelPosition, setAiChatPanelPosition] = useState<{ x: number; y: number }>(() => {
    const saved = localStorage.getItem('omniflow_ai_chat_panel_position');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        // 默认位置：右下角按钮的左边
        if (typeof window !== 'undefined') {
          return {
            x: window.innerWidth - 400 - 104,
            y: window.innerHeight - 500 - 24
          };
        }
      }
    }
    // 默认位置：右下角按钮的左边（按钮在 right-6 bottom-6，即距离右边和底部24px，按钮宽度56px）
    // 对话框应该在按钮左边，距离右边 24 + 56 + 24 = 104px
    if (typeof window !== 'undefined') {
      return {
        x: window.innerWidth - 400 - 104,
        y: window.innerHeight - 500 - 24
      };
    }
    return { x: 0, y: 0 };
  });

  const [aiChatPanelSize, setAiChatPanelSize] = useState<{ width: number; height: number }>(() => {
    const saved = localStorage.getItem('omniflow_ai_chat_panel_size');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return { width: 400, height: 500 };
      }
    }
    return { width: 400, height: 500 };
  });

  // 右侧面板高度比例（NodeConfigPanel 占用的比例，0-1之间）
  const [rightPanelSplitRatio, setRightPanelSplitRatio] = useState<number>(() => {
    const saved = localStorage.getItem('omniflow_right_panel_split_ratio');
    return saved ? parseFloat(saved) : 0.5; // 默认各占50%
  });

  return {
    // Modal states
    showCloneVoiceModal,
    setShowCloneVoiceModal,
    showAudioEditor,
    setShowAudioEditor,
    showVideoEditor,
    setShowVideoEditor,
    expandedOutput,
    setExpandedOutput,
    isEditingResult,
    setIsEditingResult,
    tempEditValue,
    setTempEditValue,

    // Menu states
    showReplaceMenu,
    setShowReplaceMenu,
    showOutputQuickAdd,
    setShowOutputQuickAdd,
    showModelSelect,
    setShowModelSelect,
    showVoiceSelect,
    setShowVoiceSelect,

        // Panel states
        sidebarCollapsed,
        setSidebarCollapsed,
        resultsCollapsed,
        setResultsCollapsed,

        // AI Chat states
        isAIChatOpen,
        setIsAIChatOpen,
        isAIChatCollapsed,
        setIsAIChatCollapsed,

        // NodeConfigPanel states
        nodeConfigPanelCollapsed,
        setNodeConfigPanelCollapsed,

        // AI Chat Panel position and size
        aiChatPanelPosition,
        setAiChatPanelPosition,
        aiChatPanelSize,
        setAiChatPanelSize,

        // Right panel split ratio
        rightPanelSplitRatio,
        setRightPanelSplitRatio
      };
    };
