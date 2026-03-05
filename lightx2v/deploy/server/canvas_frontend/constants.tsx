
import { ToolDefinition, DataType, ModelDefinition } from './types';

/**
 * Check if an environment variable is set (non-empty)
 * Vite replaces process.env.XXX with JSON.stringify(value) at build time
 * - If env var is set: becomes "actual-value"
 * - If env var is not set and has || '': becomes ""
 * - If env var is not set without || '': becomes "undefined" (string)
 */
const hasEnvVar = (key: string): boolean => {
  try {
    // @ts-ignore - process.env is defined by Vite's define config
    // Vite only replaces static access like process.env.DEEPSEEK_API_KEY
    // So we need to check each key directly
    let value: string | undefined;
    switch (key) {
      case 'DEEPSEEK_API_KEY':
        value = process.env.DEEPSEEK_API_KEY;
        break;
      case 'GEMINI_API_KEY':
        value = process.env.GEMINI_API_KEY;
        break;
      case 'API_KEY':
        value = process.env.API_KEY;
        break;
      case 'PPCHAT_API_KEY':
        value = process.env.PPCHAT_API_KEY;
        break;
      case 'LIGHTX2V_URL':
        value = process.env.LIGHTX2V_URL;
        break;
      case 'LIGHTX2V_TOKEN':
        value = process.env.LIGHTX2V_TOKEN;
        break;
      case 'LIGHTX2V_CLOUD_URL':
        value = process.env.LIGHTX2V_CLOUD_URL;
        break;
      case 'LIGHTX2V_CLOUD_TOKEN':
        value = process.env.LIGHTX2V_CLOUD_TOKEN;
        break;
      default:
        // For unknown keys, try dynamic access (may not work with Vite)
        value = (process.env as any)[key];
    }

    // Check if value exists, is not empty, and is not the string "undefined"
    const result = typeof value === 'string' && value.trim() !== '' && value !== 'undefined';

    // Debug log for troubleshooting
    if (key === 'DEEPSEEK_API_KEY') {
      console.log('[hasEnvVar] DEEPSEEK_API_KEY check:', {
        key,
        valueType: typeof value,
        valueLength: value?.length || 0,
        valuePreview: value ? `${value.substring(0, 10)}...` : 'empty',
        result
      });
    }

    if (key === 'LIGHTX2V_CLOUD_TOKEN') {
      console.log('[hasEnvVar] LIGHTX2V_CLOUD_TOKEN check:', {
        key,
        valueType: typeof value,
        valueLength: value?.length || 0,
        valuePreview: value ? `${value.substring(0, 10)}...` : 'empty',
        result
      });
    }

    return result;
  } catch (error) {
    console.warn(`[hasEnvVar] Error checking ${key}:`, error);
    return false;
  }
};

/**
 * Task type to tool ID mapping
 */
const TASK_TO_TOOL_MAP: Record<string, string> = {
  't2i': 'text-to-image',
  'i2i': 'image-to-image',
  't2v': 'video-gen-text',
  'i2v': 'video-gen-image',
  'flf2v': 'video-gen-dual-frame',
  's2v': 'avatar-gen',
  'animate': 'character-swap'
};

/**
 * Get default params for a model based on task type
 */
const getDefaultParamsForTask = (task: string): any => {
  const defaultParamsMap: Record<string, any> = {
    't2i': { aspectRatio: '1:1' },
    'i2i': { aspectRatio: '1:1' },
    't2v': { aspectRatio: '16:9' },
    'i2v': { aspectRatio: '16:9' },
    'flf2v': { aspectRatio: '16:9' },
    's2v': {},
    'animate': {}
  };
  return defaultParamsMap[task] || {};
};

// 用于防止短时间内多次调用 updateLightX2VModels
let lastUpdateTime = 0;
const UPDATE_DEBOUNCE_MS = 1000; // 1秒内的重复调用会被忽略

/**
 * Update LightX2V models dynamically based on API response
 * @param models Local models from API
 * @param cloudModels Optional cloud models from API (will be added with -cloud suffix)
 */
export const updateLightX2VModels = (
  models: Array<{ task: string; model_cls: string; stage: string }>,
  cloudModels?: Array<{ task: string; model_cls: string; stage: string }>
) => {
  // 防抖：如果1秒内被多次调用，只处理最后一次
  const now = Date.now();
  if (now - lastUpdateTime < UPDATE_DEBOUNCE_MS) {
    console.log('[LightX2V] updateLightX2VModels called too frequently, skipping...');
    return;
  }
  lastUpdateTime = now;

  // Group models by task and deduplicate by model_cls
  const modelsByTask: Record<string, Map<string, { model_cls: string; stage: string; is_cloud?: boolean }>> = {};

  // Add local models
  models.forEach(model => {
    if (!modelsByTask[model.task]) {
      modelsByTask[model.task] = new Map();
    }
    // Use model_cls as key to deduplicate (case-insensitive)
    const key = model.model_cls.toLowerCase();
    if (!modelsByTask[model.task].has(key)) {
      modelsByTask[model.task].set(key, { model_cls: model.model_cls, stage: model.stage, is_cloud: false });
    }
  });

  // Add cloud models with -cloud suffix
  if (cloudModels && cloudModels.length > 0) {
    cloudModels.forEach(model => {
      if (!modelsByTask[model.task]) {
        modelsByTask[model.task] = new Map();
      }
      // For cloud models, add -cloud suffix and use as key
      const cloudModelCls = `${model.model_cls}-cloud`;
      const key = cloudModelCls.toLowerCase();
      if (!modelsByTask[model.task].has(key)) {
        modelsByTask[model.task].set(key, { model_cls: cloudModelCls, stage: model.stage, is_cloud: true });
      }
    });
  }

  // Helper function to identify LightX2V models
  const modelIdLower = (id: string) => id.toLowerCase();
  const isLightX2VModel = (id: string) => {
    const lower = modelIdLower(id);
    return lower.includes('qwen') ||
           lower.includes('wan') ||
           lower.includes('sekotalk') ||
           lower.includes('lightx2v') ||
           lower.includes('z-image') ||
           lower.includes('self-forcing') ||
           lower.includes('matrix-game');
  };

  // Update each tool's models
  Object.keys(modelsByTask).forEach(task => {
    const toolId = TASK_TO_TOOL_MAP[task];
    if (!toolId) {
      console.warn(`[LightX2V] Unknown task type: ${task}`);
      return;
    }

    const tool = TOOLS.find(t => t.id === toolId);
    if (!tool) {
      console.warn(`[LightX2V] Tool not found: ${toolId}`);
      return;
    }

    // 完全清除所有 LightX2V 模型，只保留非 LightX2V 模型（如 Gemini）
    const existingNonLightX2VModels = (tool.models || []).filter(
      (m: ModelDefinition) => !isLightX2VModel(m.id)
    );

    // Get unique model_cls values from API response for this task
    const uniqueModels = Array.from(modelsByTask[task].values());

    // Create LightX2V models from API response (deduplicated by model_cls, case-insensitive)
    const modelMap = new Map<string, ModelDefinition>();
    uniqueModels.forEach(model => {
      const modelId = model.model_cls;
      const key = modelId.toLowerCase();
      // Only add if not already in map
      if (!modelMap.has(key)) {
        const displayName = model.is_cloud
          ? `LightX2V Cloud (${model.model_cls.replace(/-cloud$/, '')})`
          : `LightX2V (${modelId})`;
        modelMap.set(key, {
          id: modelId, // 保持原始大小写（包括 -cloud 后缀）
          name: displayName,
          defaultParams: getDefaultParamsForTask(task)
        });
      }
    });
    const lightX2VModels: ModelDefinition[] = Array.from(modelMap.values());

    // Combine models: LightX2V models first, then non-LightX2V models
    // Use Map to ensure no duplicates (case-insensitive)
    const finalModelMap = new Map<string, ModelDefinition>();

    // Add LightX2V models first
    lightX2VModels.forEach(model => {
      finalModelMap.set(model.id.toLowerCase(), model);
    });

    // Add non-LightX2V models (only if not already present)
    existingNonLightX2VModels.forEach(model => {
      const key = model.id.toLowerCase();
      if (!finalModelMap.has(key)) {
        finalModelMap.set(key, model);
      }
    });

    // Update tool models
    tool.models = getFilteredModels(Array.from(finalModelMap.values()));

    console.log(`[LightX2V] Updated models for ${toolId}:`, tool.models.map((m: ModelDefinition) => m.id));
  });
};

/**
 * Get the filtered models based on available API keys
 */
const getFilteredModels = (models: ModelDefinition[]): ModelDefinition[] => {
  return models.filter(model => {
    // Map model IDs to their required environment variables
    const modelEnvMap: Record<string, string[]> = {
      'gemini-3-pro-preview': ['GEMINI_API_KEY', 'API_KEY'],
      'gemini-3-flash-preview': ['GEMINI_API_KEY', 'API_KEY'],
      'gemini-2.5-flash-image': ['GEMINI_API_KEY', 'API_KEY'],
      'gemini-2.5-flash-preview-tts': ['GEMINI_API_KEY', 'API_KEY'],
      'doubao-seed-1-6-vision-250815': ['DEEPSEEK_API_KEY'],
      'deepseek-v3-2-251201': ['DEEPSEEK_API_KEY'],
      'doubao-1-5-vision-pro-32k-250115': ['DEEPSEEK_API_KEY'],
      'ppchat-gemini-2.5-pro': ['PPCHAT_API_KEY'],
      'ppchat-gemini-3-pro-preview': ['PPCHAT_API_KEY'],
      'ppchat-gemini-2.5-flash': ['PPCHAT_API_KEY'],
      'lightx2v': ['LIGHTX2V_TOKEN', 'LIGHTX2V_CLOUD_TOKEN'],
    };
    if (model.id.startsWith('doubao-') || model.id.startsWith('deepseek-')) {
      return true;
    }

    const requiredEnvVars = modelEnvMap[model.id];
    if (!requiredEnvVars) {
      // If model is not in the map, show it by default (e.g., LightX2V models that don't need API keys)
      return true;
    }

    // Check if at least one of the required env vars is set
    return requiredEnvVars.some(key => hasEnvVar(key));
  });
};

export const TOOLS: ToolDefinition[] = [
  {
    id: 'text-input',
    name: 'Text Input',
    name_zh: '文本输入',
    category: 'Input',
    category_zh: '输入',
    description: 'Provide starting text for the workflow',
    description_zh: '为工作流提供初始文本',
    inputs: [],
    outputs: [{ id: 'out-text', type: DataType.TEXT, label: 'Text', label_zh: '文本' }],
    icon: 'Type'
  },
  {
    id: 'image-input',
    name: 'Image Input',
    name_zh: '图像输入',
    category: 'Input',
    category_zh: '输入',
    description: 'Upload one or more images as workflow input',
    description_zh: '上传一张或多张图片作为输入',
    inputs: [],
    outputs: [{ id: 'out-image', type: DataType.IMAGE, label: 'Image(s)', label_zh: '图像' }],
    icon: 'Image'
  },
  {
    id: 'audio-input',
    name: 'Audio Input',
    name_zh: '音频输入',
    category: 'Input',
    category_zh: '输入',
    description: 'Upload an audio file as workflow input',
    description_zh: '上传音频文件作为输入',
    inputs: [],
    outputs: [{ id: 'out-audio', type: DataType.AUDIO, label: 'Audio', label_zh: '音频' }],
    icon: 'Volume2'
  },
  {
    id: 'video-input',
    name: 'Video Input',
    name_zh: '视频输入',
    category: 'Input',
    category_zh: '输入',
    description: 'Upload a video file as workflow input',
    description_zh: '上传视频文件作为输入',
    inputs: [],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Video' }],
    icon: 'Video'
  },
  {
    id: 'text-generation',
    name: 'AI Chat (LLM)',
    name_zh: '文本生成 (大模型)',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Advanced reasoning and generation with multiple text outputs',
    description_zh: '基于大语言模型的高级推理和文本生成能力',
    inputs: [
      { id: 'in-text', type: DataType.TEXT, label: 'Prompt', label_zh: '提示词' },
      { id: 'in-image', type: DataType.IMAGE, label: 'Image (Optional)', label_zh: '图像（可选）' }
    ],
    outputs: [], // Dynamically managed via node.data.customOutputs
    icon: 'Cpu',
    defaultParams: {
      mode: 'basic',
      custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
    },
    models: getFilteredModels([
      {
        id: 'deepseek-v3-2-251201',
        name: 'DeepSeek V3.2',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      },
      {
        id: 'doubao-seed-1-6-vision-250815',
        name: 'Doubao Seed 1.6',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      },
      {
        id: 'ppchat-gemini-2.5-flash',
        name: 'Gemini 2.5 Flash',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      },
      {
        id: 'ppchat-gemini-3-pro-preview',
        name: 'Gemini 3 Pro',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      },
      {
        id: 'gemini-3-pro-preview',
        name: 'Gemini 3 Pro',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      },
      {
        id: 'gemini-3-flash-preview',
        name: 'Gemini 3 Flash',
        defaultParams: {
          mode: 'basic',
          custom_outputs: [{ id: 'out-text1', label: '执行结果', description: 'Main text response.' }]
        }
      }
    ])
  },
  {
    id: 'text-to-image',
    name: 'Text-to-Image',
    name_zh: '文生图',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Generate images from text descriptions using Gemini or LightX2V',
    description_zh: '根据文本描述生成图像',
    inputs: [{ id: 'in-text', type: DataType.TEXT, label: 'Prompt' }],
    outputs: [{ id: 'out-image', type: DataType.IMAGE, label: 'Image' }],
    icon: 'Sparkles',
    defaultParams: {
      aspectRatio: '1:1'
    },
    models: getFilteredModels([
      {
        id: 'gemini-2.5-flash-image',
        name: 'Gemini (Flash Image)',
        defaultParams: {
          aspectRatio: '1:1'
        }
      }
      // LightX2V models will be added dynamically via updateLightX2VModels
    ])
  },
  {
    id: 'image-to-image',
    name: 'Image-to-Image',
    name_zh: '图生图',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Edit or transform images with text using Gemini or LightX2V',
    description_zh: '通过文本编辑或转换图像',
    inputs: [
      { id: 'in-image', type: DataType.IMAGE, label: 'Reference Image', label_zh: '参考图' },
      { id: 'in-text', type: DataType.TEXT, label: 'Edit Prompt', label_zh: '编辑提示词' }
    ],
    outputs: [{ id: 'out-image', type: DataType.IMAGE, label: 'Result', label_zh: '结果' }],
    icon: 'Palette',
    defaultParams: {
      aspectRatio: '1:1'
    },
    models: getFilteredModels([
      {
        id: 'gemini-2.5-flash-image',
        name: 'Gemini (Flash Image)',
        defaultParams: {
          aspectRatio: '1:1'
        }
      }
      // LightX2V models will be added dynamically via updateLightX2VModels
    ])
  },
  {
    id: 'tts',
    name: 'TTS (Speech)',
    name_zh: '语音合成 (TTS)',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Text-to-speech conversion',
    description_zh: '文本转语音（多音色支持）',
    inputs: [
      { id: 'in-text', type: DataType.TEXT, label: 'TTS Text', label_zh: 'TTS 文本' },
      { id: 'in-context-tone', type: DataType.TEXT, label: 'Context & Tone (Opt)', label_zh: '语境与语调（可选）' }
    ],
    outputs: [{ id: 'out-audio', type: DataType.AUDIO, label: 'Audio', label_zh: '音频' }],
    icon: 'Volume2',
    defaultParams: {
      model: 'lightx2v'
    },
    models: getFilteredModels([
      {
        id: 'lightx2v',
        name: 'LightX2V TTS',
        defaultParams: {
          voiceType: 'zh_female_vv_uranus_bigtts',
          emotionScale: 3,
          speechRate: 0,
          pitch: 0,
          loudnessRate: 0,
          resourceId: 'seed-tts-2.0'
        }
      },
      {
        id: 'gemini-2.5-flash-preview-tts',
        name: 'Gemini 2.5 TTS',
        defaultParams: {
          voice: 'Kore'
        }
      }
    ])
  },
  {
    id: 'lightx2v-voice-clone',
    name: 'Voice Clone',
    name_zh: '音色克隆',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Clone voice from audio and generate TTS with cloned voice',
    description_zh: '生成你的专属角色音色',
    inputs: [
      { id: 'in-tts-text', type: DataType.TEXT, label: 'TTS Text', label_zh: 'TTS 文本' }
    ],
    outputs: [{ id: 'out-audio', type: DataType.AUDIO, label: 'Audio', label_zh: '音频' }],
    icon: 'Mic',
    defaultParams: {
      style: '正常',
      speed: 1.0,
      volume: 0,
      pitch: 0,
      language: 'ZH_CN'
    },
    models: []
  },
  {
    id: 'video-gen-text',
    name: 'Text-to-Video',
    name_zh: '文生视频',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Generate video from text',
    description_zh: '根据文本生成视频',
    inputs: [{ id: 'in-text', type: DataType.TEXT, label: 'Prompt', label_zh: '提示词' }],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Video', label_zh: '视频' }],
    icon: 'Video',
    defaultParams: {
      aspectRatio: '16:9'
    },
    models: [
      // LightX2V models will be added dynamically via updateLightX2VModels
    ]
  },
  {
    id: 'video-gen-image',
    name: 'Image-to-Video',
    name_zh: '图生视频',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Generate video from a starting image',
    description_zh: '从首帧图像生成视频',
    inputs: [
      { id: 'in-image', type: DataType.IMAGE, label: 'Start Frame', label_zh: '起始帧' },
      { id: 'in-text', type: DataType.TEXT, label: 'Motion Prompt', label_zh: '动作提示词' }
    ],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Video', label_zh: '视频' }],
    icon: 'Clapperboard',
    defaultParams: {
      aspectRatio: '16:9'
    },
    models: [
      // LightX2V models will be added dynamically via updateLightX2VModels
    ]
  },
  {
    id: 'video-gen-dual-frame',
    name: 'Start & End Frame Video',
    name_zh: '首尾帧视频',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Generate video with start and end frame constraints (Wan 2.2)',
    description_zh: '通过首尾两张图像生成过渡视频',
    inputs: [
      { id: 'in-image-start', type: DataType.IMAGE, label: 'Start Frame', label_zh: '起始帧' },
      { id: 'in-image-end', type: DataType.IMAGE, label: 'End Frame', label_zh: '结束帧' },
      { id: 'in-text', type: DataType.TEXT, label: 'Prompt (Optional)', label_zh: '提示词（可选）' }
    ],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Video', label_zh: '视频' }],
    icon: 'FastForward',
    defaultParams: {
      aspectRatio: '16:9'
    },
    models: [
      // LightX2V models will be added dynamically via updateLightX2VModels
    ]
  },
  {
    id: 'avatar-gen',
    name: 'Digital Avatar (Lip Sync)',
    name_zh: '数字人 (口型对齐)',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Speech-to-Video talking avatar powered by LightX2V SekoTalk',
    description_zh: '根据音频生成数字人对口型视频',
    inputs: [
      { id: 'in-image', type: DataType.IMAGE, label: 'Portrait Image', label_zh: '人像图' },
      { id: 'in-audio', type: DataType.AUDIO, label: 'Voice Audio', label_zh: '语音' },
      { id: 'in-text', type: DataType.TEXT, label: 'Optional Prompt', label_zh: '可选提示词' }
    ],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Avatar Video', label_zh: '数字人视频' }],
    icon: 'UserCircle',
    defaultParams: {
      prompt: '角色根据音频说话'
    },
    models: [
      // LightX2V models will be added dynamically via updateLightX2VModels
    ]
  },
  {
    id: 'character-swap',
    name: 'Character Swap',
    name_zh: '角色替换',
    category: 'AI Model',
    category_zh: 'AI 模型',
    description: 'Replace characters in a video with a reference image',
    description_zh: '将视频中的角色替换为参考图中的形象',
    inputs: [
      { id: 'in-video', type: DataType.VIDEO, label: 'Original Video' },
      { id: 'in-image', type: DataType.IMAGE, label: 'Character Image' },
      { id: 'in-text', type: DataType.TEXT, label: 'Description' }
    ],
    outputs: [{ id: 'out-video', type: DataType.VIDEO, label: 'Swapped Video' }],
    icon: 'UserCog',
    defaultParams: {
      prompt: '将视频中的角色替换为参考图中的形象'
    },
    models: [

      // LightX2V models will be added dynamically via updateLightX2VModels
    ]
  }
];
