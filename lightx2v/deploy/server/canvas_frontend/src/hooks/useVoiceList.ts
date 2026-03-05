import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { WorkflowState } from '../../types';
import { lightX2VGetVoiceList, lightX2VGetCloneVoiceList } from '../../services/geminiService';

interface LightX2VConfig {
  url: string;
  token: string;
}

export const useVoiceList = (workflow: WorkflowState | null, selectedNodeId: string | null, getLightX2VConfig: (workflow: WorkflowState | null) => LightX2VConfig) => {
  const [lightX2VVoiceList, setLightX2VVoiceList] = useState<{ voices?: any[]; emotions?: string[]; languages?: any[] } | null>(null);
  const [loadingVoiceList, setLoadingVoiceList] = useState(false);
  const voiceListLoadedRef = useRef<string>(''); // Track which URL+token combo has been loaded
  const [voiceSearchQuery, setVoiceSearchQuery] = useState('');
  const [showVoiceFilter, setShowVoiceFilter] = useState(false);
  const [voiceFilterGender, setVoiceFilterGender] = useState<string>('all');

  const [cloneVoiceList, setCloneVoiceList] = useState<any[]>([]);
  const [loadingCloneVoiceList, setLoadingCloneVoiceList] = useState(false);
  const cloneVoiceListLoadedRef = useRef<string>('');

  // Helper function to check if voice is female
  const isFemaleVoice = useCallback((voiceType: string) => {
    return voiceType?.toLowerCase().includes('female') || false;
  }, []);

  // Filter voices based on search and filters
  const filteredVoices = useMemo(() => {
    if (!lightX2VVoiceList?.voices) return [];

    let filtered = lightX2VVoiceList.voices;

    // Filter by search query
    if (voiceSearchQuery.trim()) {
      const query = voiceSearchQuery.toLowerCase();
      filtered = filtered.filter((voice: any) =>
        (voice.name || voice.voice_name || voice.voice_type || '').toLowerCase().includes(query) ||
        (voice.voice_type || '').toLowerCase().includes(query)
      );
    }

    // Filter by gender
    if (voiceFilterGender !== 'all') {
      filtered = filtered.filter((voice: any) =>
        voice.gender === voiceFilterGender ||
        (voiceFilterGender === 'female' && isFemaleVoice(voice.voice_type)) ||
        (voiceFilterGender === 'male' && !isFemaleVoice(voice.voice_type))
      );
    }

    return filtered;
  }, [lightX2VVoiceList, voiceSearchQuery, voiceFilterGender, isFemaleVoice]);

  // Load voice list when TTS node is selected and model is lightx2v
  useEffect(() => {
    const loadVoiceList = async () => {
      if (!selectedNodeId || !workflow) return;
      const node = workflow.nodes.find(n => n.id === selectedNodeId);
      if (!node || node.tool_id !== 'tts') return;

      // Only load voice list if model is lightx2v
      const isLightX2V = node.data.model === 'lightx2v' || node.data.model?.startsWith('lightx2v');
      if (!isLightX2V) {
        // Reset voice list when switching away from lightx2v
        setLightX2VVoiceList(null);
        setVoiceSearchQuery('');
        setShowVoiceFilter(false);
        setVoiceFilterGender('all');
        voiceListLoadedRef.current = '';
        return;
      }

      // Get config from env vars or apiClient
      const config = getLightX2VConfig(workflow);

      // Check if we have required config
      // When using apiClient, url can be empty (relative path) and token can be empty (uses main app's JWT)
      const apiClient = (window as any).__API_CLIENT__;
      const hasApiClient = !!apiClient;

      // If not using apiClient, we need both url and token
      // If using apiClient, we can proceed with empty url (relative path)
      if (!hasApiClient && (!config.url || !config.token)) {
        console.warn('[LightX2V] Missing URL or token for voice list');
        return;
      }

      // If using apiClient but no url, that's fine (will use relative path)
      // But we still need to check if we have a way to authenticate
      if (hasApiClient && !config.url && !config.token) {
        // This is OK - apiClient will use main app's JWT token
        // But we should still have some way to identify the request
      }

      // Create a key to track if we've loaded for this URL+token combination
      const loadKey = `${config.url}:${config.token}`;

      // Don't reload if already loaded for this combination and not currently loading
      if (voiceListLoadedRef.current === loadKey && !loadingVoiceList && voiceListLoadedRef.current !== '') return;

      setLoadingVoiceList(true);
      try {
        const voiceData = await lightX2VGetVoiceList(config.url, config.token);
        setLightX2VVoiceList(voiceData);
        voiceListLoadedRef.current = loadKey;
      } catch (error: any) {
        console.error('[LightX2V] Failed to load voice list:', error);
        setLightX2VVoiceList(null);
        voiceListLoadedRef.current = '';
      } finally {
        setLoadingVoiceList(false);
      }
    };

    loadVoiceList();
  }, [selectedNodeId, workflow?.nodes.find(n => n.id === selectedNodeId)?.data?.model, loadingVoiceList]);

  // Load clone voice list when voice clone node is selected
  useEffect(() => {
    const loadCloneVoiceList = async () => {
      if (!selectedNodeId || !workflow) return;
      const node = workflow.nodes.find(n => n.id === selectedNodeId);
      if (!node || node.tool_id !== 'lightx2v-voice-clone') {
        // Reset clone voice list when switching away from voice clone node
        setCloneVoiceList([]);
        cloneVoiceListLoadedRef.current = '';
        return;
      }

      const config = getLightX2VConfig(workflow);

      // Check if we have required config
      // When using apiClient, url can be empty (relative path) and token can be empty (uses main app's JWT)
      const apiClient = (window as any).__API_CLIENT__;
      const hasApiClient = !!apiClient;

      // If not using apiClient, we need both url and token
      // If using apiClient, we can proceed with empty url (relative path)
      if (!hasApiClient && (!config.url || !config.token)) {
        console.warn('[LightX2V] Missing URL or token for clone voice list');
        return;
      }

      const loadKey = `${config.url}:${config.token}`;
      if (cloneVoiceListLoadedRef.current === loadKey && !loadingCloneVoiceList && cloneVoiceListLoadedRef.current !== '') return;

      setLoadingCloneVoiceList(true);
      try {
        const cloneList = await lightX2VGetCloneVoiceList(config.url, config.token);
        setCloneVoiceList(Array.isArray(cloneList) ? cloneList : []);
        cloneVoiceListLoadedRef.current = loadKey;
      } catch (error: any) {
        console.error('[LightX2V] Failed to load clone voice list:', error);
        setCloneVoiceList([]);
        cloneVoiceListLoadedRef.current = '';
      } finally {
        setLoadingCloneVoiceList(false);
      }
    };

    loadCloneVoiceList();
  }, [selectedNodeId, workflow?.nodes.find(n => n.id === selectedNodeId)?.tool_id, loadingCloneVoiceList]);

  return {
    lightX2VVoiceList,
    loadingVoiceList,
    cloneVoiceList,
    loadingCloneVoiceList,
    voiceSearchQuery,
    setVoiceSearchQuery,
    showVoiceFilter,
    setShowVoiceFilter,
    voiceFilterGender,
    setVoiceFilterGender,
    filteredVoices,
    isFemaleVoice,
    // Reset functions
    resetVoiceList: useCallback(() => {
      setLightX2VVoiceList(null);
      voiceListLoadedRef.current = '';
      setVoiceSearchQuery('');
      setShowVoiceFilter(false);
      setVoiceFilterGender('all');
    }, []),
    resetCloneVoiceList: useCallback(() => {
      setCloneVoiceList([]);
      cloneVoiceListLoadedRef.current = '';
    }, [])
  };
};
