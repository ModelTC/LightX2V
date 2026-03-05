import { useCallback } from 'react';
import { WorkflowState, NodeStatus } from '../../types';
import { TOOLS } from '../../constants';
import { deepseekChat, lightX2VGetVoiceList, lightX2VGetCloneVoiceList } from '../../services/geminiService';
import { useTranslation, Language } from '../i18n/useTranslation';

interface UseAIGenerateWorkflowProps {
  workflow: WorkflowState | null;
  setWorkflow: (workflow: WorkflowState | null) => void;
  setCurrentView: (view: 'DASHBOARD' | 'EDITOR') => void;
  getLightX2VConfig: (workflow: WorkflowState | null) => { url: string; token: string };
  resetView: () => void;
  lang: Language;
}

export const useAIGenerateWorkflow = ({
  workflow,
  setWorkflow,
  setCurrentView,
  getLightX2VConfig,
  resetView,
  lang
}: UseAIGenerateWorkflowProps) => {
  const { t } = useTranslation(lang);

  const generateToolsDescription = useCallback(() => {
    const toolsInfo = TOOLS.map(tool => {
      const inputs = tool.inputs.map(inp => `${inp.label} (${inp.id}: ${inp.type})`).join(', ');
      const outputs = tool.outputs.map(out => `${out.label} (${out.id}: ${out.type})`).join(', ');
      const models = tool.models?.map(m => `${m.name} (${m.id})`).join(', ') || 'N/A';
      return `- ${tool.name} (${tool.id}): ${tool.description_zh || tool.description}
  Inputs: ${inputs || 'None'}
  Outputs: ${outputs || 'None'}
  Models: ${models}
  Category: ${tool.category_zh || tool.category}`;
    }).join('\n\n');

    return `Available Tools:\n\n${toolsInfo}`;
  }, []);

  const getVoiceListForAI = useCallback(async (description: string): Promise<string> => {
    const descLower = description.toLowerCase();
    const needsTTS = descLower.includes('tts') || descLower.includes('语音') || descLower.includes('音色') || descLower.includes('voice');
    const needsClone = descLower.includes('clone') || descLower.includes('克隆') || descLower.includes('音色克隆');

    if (!needsTTS && !needsClone) return '';

    const config = getLightX2VConfig(workflow);
    if (!config.url || !config.token) return '';

    let voiceInfo = '';

    try {
      // Get TTS voice list
      const voiceList = await lightX2VGetVoiceList(config.url, config.token);
      if (voiceList.voices && voiceList.voices.length > 0) {
        const topVoices = voiceList.voices.slice(0, 10).map((v: any) =>
          `- ${v.name || v.voice_name || v.voice_type} (${v.voice_type}): ${v.gender || 'unknown'}, version ${v.version || 'N/A'}, resource_id: ${v.resource_id || 'N/A'}`
        ).join('\n');
        voiceInfo += `\n\nAvailable TTS Voices (first 10):\n${topVoices}`;
      }

      // Get clone voice list if needed
      if (needsClone) {
        const cloneList = await lightX2VGetCloneVoiceList(config.url, config.token);
        if (cloneList && cloneList.length > 0) {
          const topClone = cloneList.slice(0, 10).map((v: any) =>
            `- ${v.name || v.speaker_id} (speaker_id: ${v.speaker_id})`
          ).join('\n');
          voiceInfo += `\n\nAvailable Cloned Voices (first 10):\n${topClone}`;
        }
      }
    } catch (error: any) {
      console.warn('[AI Workflow] Failed to load voice list for AI:', error);
    }

    return voiceInfo;
  }, [workflow, getLightX2VConfig]);

  const generateWorkflowWithAI = useCallback(async (description: string) => {
    try {
      // Get tools description
      const toolsDesc = generateToolsDescription();

      // Get voice list if needed
      const voiceInfo = await getVoiceListForAI(description);

      // Build AI prompt
      const prompt = `You are a workflow design assistant. The user wants to create a workflow based on this description:

"${description}"

${toolsDesc}${voiceInfo}

Please generate a workflow in JSON format with the following structure:
{
  "nodes": [
    {
      "id": "node-1",
      "toolId": "text-input",
      "x": 100,
      "y": 200,
      "data": { "value": "..." }
    },
    ...
  ],
  "connections": [
    {
      "source_node_id": "node-1",
      "source_port_id": "out-text",
      "target_node_id": "node-2",
      "target_port_id": "in-text"
    },
    ...
  ],
  "name": "Generated Workflow Name"
}

Requirements:
1. Each node must have a unique id (like "node-1", "node-2", etc.)
2. toolId must match one of the available tool IDs
3. For tools with models, set data.model to one of the available model IDs
4. For TTS nodes with LightX2V, you can use any voice_type from the available voices list
5. For voice clone nodes, you can use any speaker_id from the cloned voices list
6. Position nodes with reasonable x, y coordinates (spacing: x increments by 400, y increments by 200)
7. Connect nodes logically - source outputs must match target input types
8. Use appropriate default values in data (e.g., aspectRatio for video tools)
9. Provide a descriptive name for the workflow

IMPORTANT - Smart Default Values:
Based on the user's description, intelligently fill in default values for nodes to reduce user editing work:
- For text-input nodes: Set data.value to appropriate text based on user requirements
- For image-to-image nodes: Set data.value (prompt field) to transformation instructions based on user description (e.g., if user wants "cartoon style", use prompt like "Transform the image into cartoon style, maintaining character consistency")
- For text-to-image nodes: Set data.value to image generation prompts based on user requirements
- For video nodes: Set data.value to appropriate motion/camera prompts based on user description
- For TTS nodes: Set data.value (text field) to appropriate text based on user requirements
- For AI chat nodes: Set customInstruction to appropriate system instructions based on user requirements
- Analyze the user's description carefully and extract the specific transformation, style, content, or operation they want, then set the corresponding node's data values accordingly
- The goal is to minimize the need for users to manually edit node values after workflow generation

Examples:
- If user wants "cartoon style transformation": image-to-image node should have data.value like "Transform into cartoon style, maintain character features"
- If user wants "portrait generation": text-to-image node should have data.value like "A detailed portrait photo"
- If user wants "voice narration": TTS node should have data.value with appropriate narration text
- Extract specific requirements from user description and pre-fill node values accordingly

Output ONLY the JSON, no additional text or markdown.`;

      // Call AI to generate workflow using chat completions API with JSON mode
      const messages = [
        {
          role: 'system' as const,
          content: 'You are a workflow design assistant. Always respond with valid JSON only, no additional text or markdown.'
        },
        {
          role: 'user' as const,
          content: prompt
        }
      ];

      const workflowJsonStr = await deepseekChat(
        messages,
        'deepseek-v3-2-251201',
        'json_object'
      );

      // Parse the JSON response
      let workflowData: any;
      try {
        // Ensure we have a string
        const jsonStr = typeof workflowJsonStr === 'string' ? workflowJsonStr : JSON.stringify(workflowJsonStr);

        // Try to extract JSON from markdown code blocks if present (fallback)
        let parsedStr = jsonStr;
        const jsonMatch = jsonStr.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/) || jsonStr.match(/(\{[\s\S]*\})/);
        if (jsonMatch) {
          parsedStr = jsonMatch[1];
        }

        workflowData = JSON.parse(parsedStr);
      } catch (parseError: any) {
        console.error('[AI Workflow] JSON parse error:', parseError);
        console.error('[AI Workflow] Response was:', workflowJsonStr);
        throw new Error(`Failed to parse AI response as JSON: ${parseError.message || parseError}`);
      }

      // Validate and create workflow
      if (!workflowData.nodes || !Array.isArray(workflowData.nodes)) {
        throw new Error('Invalid workflow: nodes array is required');
      }

      // Create new workflow
      const newFlow: WorkflowState = {
        id: `flow-${Date.now()}`,
        name: workflowData.name || t('untitled'),
        nodes: workflowData.nodes.map((n: any, idx: number) => {
          const tool = TOOLS.find(t => t.id === n.tool_id);
          const defaultData: Record<string, any> = { ...(n.data || {}) };

          // Set default model if tool has models
          if (tool?.models && tool.models.length > 0 && !defaultData.model) {
            defaultData.model = tool.models[0].id;
          }

          // Set default aspectRatio for video tools
          if (tool?.id.includes('video-gen') && !defaultData.aspectRatio) {
            defaultData.aspectRatio = "16:9";
          }

          // Set default value for text-input
          if (tool?.id === 'text-input' && defaultData.value === undefined) {
            defaultData.value = "";
          }

          return {
            id: n.id || `node-${Date.now()}-${idx}`,
            tool_id: n.tool_id,
            x: n.x || (idx * 400),
            y: n.y || (idx % 3 * 200),
            status: NodeStatus.IDLE,
            data: defaultData
          };
        }),
        connections: (workflowData.connections || []).map((c: any, idx: number) => ({
          id: c.id || `conn-${Date.now()}-${idx}`,
          source_node_id: c.source_node_id ?? c.sourceNodeId,
          source_port_id: c.source_port_id ?? c.sourcePortId,
          target_node_id: c.target_node_id ?? c.targetNodeId,
          target_port_id: c.target_port_id ?? c.targetPortId
        })),
        isDirty: true,
        isRunning: false,
        globalInputs: {},
        updatedAt: Date.now()
      };

      // Set workflow and open editor
      setWorkflow(newFlow);
      setCurrentView('EDITOR');

      // Reset view to show all nodes
      setTimeout(() => {
        resetView();
      }, 100);

      return newFlow;
    } catch (error: any) {
      console.error('[AI Workflow] Generation failed:', error);
      throw error;
    }
  }, [generateToolsDescription, getVoiceListForAI, setWorkflow, setCurrentView, resetView, t]);

  return {
    generateWorkflowWithAI
  };
};
