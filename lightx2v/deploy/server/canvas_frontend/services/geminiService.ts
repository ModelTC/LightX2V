
import { GoogleGenAI, Modality, Type } from "@google/genai";
import { isStandalone } from '../src/config/runtimeMode';
import { apiRequest, getApiBaseUrl } from '../src/utils/apiClient';
import { getWorkflowFileByFileId } from '../src/utils/workflowFileManager';
import { toLightX2VResultRef, type LightX2VResultRef } from '../src/utils/resultRef';

const VOLC_RESPONSES_URL = 'https://ark.cn-beijing.volces.com/api/v3/responses';

/** Call Volc /responses: direct when standalone, backend proxy (with user auth) when not. */
async function volcResponsesRequest(requestBody: any, signal?: AbortSignal): Promise<Response> {
  if (isStandalone()) {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey?.trim()) {
      throw new Error("API key is required. Please set DEEPSEEK_API_KEY environment variable.");
    }
    return fetch(VOLC_RESPONSES_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(requestBody),
      signal
    });
  }
  const base = getApiBaseUrl();
  return apiRequest(`${base}/api/v1/llm/volc/responses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
    signal
  });
}

export interface OutputField {
  id: string;
  description: string;
}

/**
 * Helper to wrap raw 16-bit mono PCM data in a WAV (RIFF) header.
 * Necessary because external APIs like LightX2V often expect a valid audio container.
 */
const wrapPcmInWav = (base64Pcm: string, sampleRate = 24000): string => {
  const pcmData = atob(base64Pcm);
  const len = pcmData.length;
  const buffer = new ArrayBuffer(44 + len);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  /* RIFF identifier */
  writeString(0, 'RIFF');
  /* file length */
  view.setUint32(4, 32 + len, true);
  /* RIFF type */
  writeString(8, 'WAVE');
  /* format chunk identifier */
  writeString(12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw PCM) */
  view.setUint16(20, 1, true);
  /* channel count (mono) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(36, 'data');
  /* data chunk length */
  view.setUint32(40, len, true);

  /* Write actual PCM data */
  for (let i = 0; i < len; i++) {
    view.setUint8(44 + i, pcmData.charCodeAt(i));
  }

  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

export const geminiText = async (
  prompt: string,
  useSearch = false,
  mode = 'basic',
  customInstruction?: string,
  model = 'gemini-3-pro-preview',
  outputFields?: OutputField[],
  imageInput?: string | string[] | any[]
): Promise<any> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const systemInstructions: Record<string, string> = {
    basic: "You are a helpful and versatile AI assistant. Provide clear, accurate, and direct answers.",
    enhance: "You are a Prompt Engineering Expert. Enhance the user's input into a detailed prompt. Output ONLY the enhanced prompt.",
    enhance_image: "Expand the user's input into a highly detailed image generation prompt. Output ONLY the prompt.",
    enhance_video: "Expand the user's input into a cinematic video prompt. Describe camera and lighting. Output ONLY the prompt.",
    enhance_tts: "Transform the input into a natural narration script. Output ONLY the text.",
    summarize: "Extract core info into a concise summary.",
    polish: "Refine text for clarity and tone.",
  };

  const baseInstruction = mode === 'custom' && customInstruction
    ? customInstruction
    : (systemInstructions[mode] || systemInstructions.basic);

  const hasMultipleOutputs = outputFields && outputFields.length > 0;
  const outputKeys = outputFields?.map(f => f.id) || [];

  const finalInstruction = hasMultipleOutputs
    ? `${baseInstruction}\n\nIMPORTANT: You MUST generate content for each field: ${outputKeys.join(', ')}.`
    : baseInstruction;

  let contents: string | { parts: any[] };
  if (imageInput) {
    const inputs = Array.isArray(imageInput) ? imageInput : [imageInput];
    const flatInputs = inputs.flat().filter((img: any) => img && typeof img === 'string');
    const parts: any[] = [{ text: prompt }];
    flatInputs.forEach((img: string) => {
      const mimeMatch = img.startsWith('data:') ? img.match(/^data:([^;]+);/) : null;
      const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
      const data = img.includes(',') ? img.split(',')[1] : img;
      parts.push({ inlineData: { data, mimeType } });
    });
    contents = { parts };
  } else {
    contents = prompt;
  }

  const response = await ai.models.generateContent({
    model: model,
    contents,
    config: {
      systemInstruction: finalInstruction,
      tools: useSearch ? [{ googleSearch: {} }] : undefined,
      responseMimeType: hasMultipleOutputs ? "application/json" : "text/plain",
      responseSchema: hasMultipleOutputs ? {
        type: Type.OBJECT,
        properties: (outputFields || []).reduce((acc, field) => ({
          ...acc,
          [field.id]: { type: Type.STRING, description: field.description || field.id }
        }), {}),
        required: outputKeys
      } : undefined
    }
  });

  const text = response.text || "";
  if (hasMultipleOutputs) {
    try {
      const parsed = JSON.parse(text);
      outputKeys.forEach(key => { if (!(key in parsed)) parsed[key] = "..."; });
      return parsed;
    } catch (e) {
      const fallback: Record<string, string> = {};
      outputKeys.forEach((key, i) => fallback[key] = i === 0 ? text : "...");
      return fallback;
    }
  }
  return text;
};

export const geminiImage = async (prompt: string, imageInput?: string | string[] | any[], aspectRatio = "1:1", model = 'gemini-2.5-flash-image'): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const parts: any[] = [{ text: prompt }];

  if (imageInput) {
    const inputs = Array.isArray(imageInput) ? imageInput : [imageInput];
    const flatInputs = inputs.flat().filter(img => img && typeof img === 'string');
    flatInputs.forEach(img => {
      const data = img.includes(',') ? img.split(',')[1] : img;
      parts.push({ inlineData: { data: data, mimeType: 'image/png' } });
    });
  }

  const response = await ai.models.generateContent({
    model: model,
    contents: { parts: parts },
    config: { imageConfig: { aspectRatio: aspectRatio as any } }
  });

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.inlineData) return `data:image/png;base64,${part.inlineData.data}`;
  }
  throw new Error("No image generated");
};

export const geminiSpeech = async (text: string, voice = 'Kore', model = "gemini-2.5-flash-preview-tts", tone?: string): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const finalPrompt = tone ? `Style: ${tone}\nText: ${text}` : text;
  const response = await ai.models.generateContent({
    model: model,
    contents: [{ parts: [{ text: finalPrompt }] }],
    config: {
      responseModalities: [Modality.AUDIO],
      speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
    },
  });
  const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!base64Audio) throw new Error("No audio generated");
  return base64Audio;
};

export const geminiVideo = async (prompt: string, imageBase64?: string, aspectRatio = "16:9", resolution = "720p", refVideo?: any, model = 'veo-3.1-fast-generate-preview'): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const cleanedImage = imageBase64?.split(',')[1] || imageBase64;
  let operation = await ai.models.generateVideos({
    model: model,
    prompt: prompt,
    image: cleanedImage ? { imageBytes: cleanedImage, mimeType: 'image/png' } : undefined,
    video: refVideo,
    config: { numberOfVideos: 1, resolution: resolution as any, aspectRatio: aspectRatio as any }
  });
  while (!operation.done) {
    await new Promise(res => setTimeout(res, 10000));
    operation = await ai.operations.getVideosOperation({operation: operation});
  }
  const link = operation.response?.generatedVideos?.[0]?.video?.uri;
  if (!link) throw new Error("Video failed");
  return `${link}&key=${process.env.API_KEY}`;
};

/**
 * ============================================================================
 * LightX2V Unified API Client
 * ============================================================================
 * 统一的 LightX2V API 请求函数，处理 apiClient 和直接 fetch 两种模式
 */

interface LightX2VRequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: string | FormData;
  accept?: string;
}

interface LightX2VRequestConfig {
  baseUrl: string;
  token: string;
  endpoint: string;
  options?: LightX2VRequestOptions;
  requireToken?: boolean; // 是否要求 token（使用 apiClient 时可以为 false）
}

/**
 * 统一的 LightX2V API 请求函数
 * @param config 请求配置
 * @returns Response 对象
 */
async function lightX2VRequest(config: LightX2VRequestConfig): Promise<Response> {
  const { baseUrl, token, endpoint, options = {}, requireToken = true } = config;

  // 验证 baseUrl 格式
  if (baseUrl && baseUrl.trim() && !baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
    throw new Error("Base URL must be a valid HTTP/HTTPS URL or empty string for relative path");
  }

  // 判断是否使用 apiClient（baseUrl 为空时使用）
  const useApiClient = !baseUrl || !baseUrl.trim();

  // 使用 apiClient 时，不需要检查 token（后端会使用环境变量中的 LIGHTX2V_TOKEN）
  if (!useApiClient && requireToken && (!token || !token.trim())) {
    throw new Error("Access Token is required for LightX2V");
  }

  // 构建 URL
  const normalizedBaseUrl = useApiClient ? '' : baseUrl.replace(/\/$/, '');
  const url = useApiClient
    ? endpoint.startsWith('/') ? endpoint : `/${endpoint}`
    : `${normalizedBaseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  // 构建 headers
  const defaultHeaders: Record<string, string> = {
    'Accept': options.accept || 'application/json'
  };

  if (options.method === 'POST' || options.method === 'PUT' || options.method === 'PATCH') {
    if (!(options.body instanceof FormData)) {
      defaultHeaders['Content-Type'] = 'application/json; charset=utf-8';
    }
  }

  const headers: Record<string, string> = {
    ...defaultHeaders,
    ...(options.headers || {})
  };

  // 使用 apiClient 时，不传递 LightX2V token（apiRequest 会自动使用主应用的 JWT token）
  // 不使用 apiClient 时，传递 LightX2V token
  if (!useApiClient && requireToken) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // 执行请求
  if (useApiClient) {
    return await apiRequest(url, {
      method: options.method || 'GET',
      headers,
      body: options.body
    });
  } else {
    return await fetch(url, {
      method: options.method || 'GET',
      headers,
      body: options.body
    });
  }
}

/**
 * 统一的错误处理函数
 */
async function handleLightX2VError(response: Response, context: string): Promise<never> {
  let errorMessage = `LightX2V ${context} Failed (${response.status})`;
  try {
    const errText = await response.text();
    if (errText.trim().startsWith('{') || errText.trim().startsWith('[')) {
      try {
        const errorData = JSON.parse(errText);
        errorMessage = errorData.error || errorData.message || errorMessage;
        if (errorData.detail) errorMessage += `: ${errorData.detail}`;
      } catch {
        errorMessage = errText.trim() || errorMessage;
      }
    } else {
      errorMessage = errText.trim() || errorMessage;
    }
  } catch (e: any) {
    errorMessage = e.message || errorMessage;
  }
  console.error(`[LightX2V] ${context} error: ${errorMessage}`, {
    status: response.status,
    url: response.url
  });
  throw new Error(errorMessage);
}

/**
 * ============================================================================
 * LightX2V API Functions
 * ============================================================================
 */

/** 本地 workflow_output 模式：提交 (workflow_id, node_id, port_id) 元组，后端解析 */
export type WorkflowRefsPayload = {
  workflowId: string;
  prompt?: [string, string, string];
  input_image?: [string, string, string] | [string, string, string][];
  input_audio?: [string, string, string];
  input_video?: [string, string, string];
  input_last_frame?: [string, string, string];
};

/**
 * LightX2V Video/Image Service Integration
 * Generalized to handle T2V, I2V, S2V, T2I, and I2I
 */
export const lightX2VTask = async (
  workflow_id: string,
  node_id: string,
  port_id: string,
  baseUrl: string,
  token: string,
  task: string,
  modelCls: string,
  prompt: string,
  inputImage?: string | string[],
  inputAudio?: string,
  lastFrame?: string,
  outputName = "output_video",
  aspectRatio?: string,
  inputVideo?: string,
  onTaskId?: (taskId: string) => void,
  abortSignal?: AbortSignal,
  workflowRefs?: WorkflowRefsPayload
): Promise<LightX2VResultRef> => {
  // Check if this is a cloud model (ends with -cloud)
  // If so, use cloud config instead of provided baseUrl/token
  const isCloud = modelCls.endsWith('-cloud');
  let actualBaseUrl = baseUrl;
  let actualToken = token;

  if (isCloud) {
    // Cloud model - use cloud config
    const cloudConfig = getLightX2VConfigForModel(modelCls);
    actualBaseUrl = cloudConfig.url;
    actualToken = cloudConfig.token;
    // Remove -cloud suffix for actual API call
    modelCls = modelCls.replace(/-cloud$/, '');
    console.log('[LightX2V] Using cloud backend for model:', modelCls);
  }

  console.log('[LightX2V] baseUrl:', actualBaseUrl);
  const useWorkflowRefs = !!workflowRefs && !isCloud;

  // 云端且存在 workflowId 时，将 file ref 解析为 data URL 再提交（连接后端但选云端模型时图片/音视频为 file_id）
  let resolvedInputImage = inputImage;
  let resolvedInputAudio = inputAudio;
  let resolvedLastFrame = lastFrame;
  let resolvedInputVideo = inputVideo;
  if (isCloud && workflowRefs?.workflowId) {
    const wfId = workflowRefs.workflowId;
    const isFileRef = (v: unknown): v is { file_id: string; mime_type?: string } =>
      !!v && typeof v === 'object' && !Array.isArray(v) && typeof (v as { file_id?: string }).file_id === 'string';
    if (resolvedInputImage != null) {
      if (isFileRef(resolvedInputImage)) {
        const r = await getWorkflowFileByFileId(wfId, resolvedInputImage.file_id, resolvedInputImage.mime_type, (resolvedInputImage as { ext?: string }).ext);
        if (r) resolvedInputImage = r;
      } else if (Array.isArray(resolvedInputImage)) {
        const arr = await Promise.all(resolvedInputImage.map(async (item) => {
          if (isFileRef(item)) {
            const r = await getWorkflowFileByFileId(wfId, item.file_id, item.mime_type, (item as { ext?: string }).ext);
            return r ?? item;
          }
          return item;
        }));
        const strings = arr.filter((x): x is string => typeof x === 'string');
        resolvedInputImage = strings.length > 0 ? strings : undefined;
      }
    }
    if (isFileRef(resolvedInputAudio)) {
      const r = await getWorkflowFileByFileId(wfId, resolvedInputAudio.file_id, resolvedInputAudio.mime_type, (resolvedInputAudio as { ext?: string }).ext);
      if (r) resolvedInputAudio = r;
    }
    if (isFileRef(resolvedLastFrame)) {
      const r = await getWorkflowFileByFileId(wfId, resolvedLastFrame.file_id, resolvedLastFrame.mime_type, (resolvedLastFrame as { ext?: string }).ext);
      if (r) resolvedLastFrame = r;
    }
    if (isFileRef(resolvedInputVideo)) {
      const r = await getWorkflowFileByFileId(wfId, resolvedInputVideo.file_id, resolvedInputVideo.mime_type, (resolvedInputVideo as { ext?: string }).ext);
      if (r) resolvedInputVideo = r;
    }
  }

  const formatMediaPayload = (val: string | string[] | undefined, isAudio = false) => {
    if (!val) return undefined;
    // 未解析的 file ref（对象）不能当字符串用，否则后端会收到无效 base64
    if (typeof val === 'object' && !Array.isArray(val)) return undefined;

    // Handle multiple images (array) - for i2i tasks with multiple input images
    // According to lightx2v server (utils.py:177-185), server expects:
    // - For base64: list of base64 strings (may include data:image prefix)
    // - Server will decode each and save as input_image_1, input_image_2, etc.
    if (Array.isArray(val) && val.length > 0) {
      // Process each image: extract base64 from data URLs, keep URLs as-is
      const isPathOrHttpUrl = (s: string) =>
        s.startsWith('http://') || s.startsWith('https://') || s.startsWith('./') ||
        s.startsWith('/assets') || s.startsWith('/api');

      const processedImages = val.map(img => {
        if (typeof img !== 'string') return img;
        if (isPathOrHttpUrl(img)) {
          return img;
        }
        if (img.startsWith('data:image')) {
          return img.split(',')[1];
        }
        if (img.includes(',')) {
          return img.split(',')[1];
        }
        return img;
      }).filter((x): x is string => typeof x === 'string' && x.length > 0);

      if (processedImages.length === 0) return undefined;

      // 不能仅用 startsWith('/') 判 URL：JPEG base64 以 "/9j/4AAQ" 开头，会被误判为 url 导致 Submit error
      // 必须全部为 URL 才用 type=url；任一 base64 时用 type=base64，否则后端会尝试 fetch base64 导致 "Failed to read input_image, type=url"
      const allUrls = processedImages.every(img =>
        typeof img === 'string' && isPathOrHttpUrl(img)
      );
      const type = allUrls ? "url" : "base64";

      return { type: type, data: processedImages };
    }

    // Single value handling (original logic)
    const singleVal = val as string;
    // Check if it's a URL (absolute http/https, or relative path ./ or /assets or /api)
    // 不能仅用 startsWith('/')：JPEG base64 以 "/9j/4AAQ" 开头，会被误判为 url
    const isUrl = singleVal.startsWith('http://') ||
                  singleVal.startsWith('https://') ||
                  singleVal.startsWith('./') ||
                  singleVal.startsWith('/assets') ||
                  singleVal.startsWith('/api');
    const type = isUrl ? "url" : "base64";

    // Process the data content
    let dataContent = singleVal;
    if (!isUrl) {
      dataContent = singleVal.includes(',') ? singleVal.split(',')[1] : singleVal;
      // Special handling for raw PCM audio from Gemini
      if (isAudio && !singleVal.startsWith('data:')) {
        dataContent = wrapPcmInWav(dataContent, 24000);
      }
      // 空 base64 会导致云端 "cannot identify image file"；不提交
      if (typeof dataContent !== 'string' || !dataContent.trim()) return undefined;
    }

    // Use 'data' as the field name for both types as the server error 'data'! suggests
    return { type: type, data: dataContent };
  };

  // 参考主应用的做法构建 payload；本地走 workflow_output 元组时由后端解析
  // 后端 assert len(params["prompt"]) > 0，云端提交时 prompt 不能为空字符串（否则 400 valid prompt is required）
  const promptPayload = useWorkflowRefs && workflowRefs!.prompt
    ? workflowRefs!.prompt
    : (typeof prompt === "string" && prompt.trim().length > 0 ? prompt.trim() : " ");
  const payload: any = {
    task: task,
    model_cls: modelCls,
    stage: "single_stage",
    seed: Math.floor(Math.random() * 1000000), // 参考主应用：生成随机 seed
    prompt: promptPayload
  };

  // 添加尺寸参数（参考主应用）
  if (aspectRatio) payload.aspect_ratio = aspectRatio;

  // 添加 negative_prompt（参考主应用：对于 wan2.1/wan2.2 模型）
  if (modelCls.startsWith('wan2.1') || modelCls.startsWith('wan2.2')) {
    payload.negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
  }

  if (useWorkflowRefs && workflowRefs!) {
    // 本地 workflow_output：只传元组，后端按 (workflow_id, node_id, port_id) 解析
    if (workflowRefs.input_image !== undefined) {
      payload.input_image = { type: "workflow_output", data: workflowRefs.input_image };
    }
    if (workflowRefs.input_audio !== undefined) {
      payload.input_audio = { type: "workflow_output", data: workflowRefs.input_audio };
    }
    if (workflowRefs.input_video !== undefined) {
      payload.input_video = { type: "workflow_output", data: workflowRefs.input_video };
    }
    if (workflowRefs.input_last_frame !== undefined) {
      payload.input_last_frame = { type: "workflow_output", data: workflowRefs.input_last_frame };
    }
  } else {
    // 云端或未传 workflowRefs：沿用现有格式（prompt 字符串 + 媒体 type/data）；云端时已用 resolved*（file ref 已解析为 data URL）
    if (resolvedInputImage) payload.input_image = formatMediaPayload(resolvedInputImage);
    if (resolvedInputAudio) payload.input_audio = formatMediaPayload(resolvedInputAudio, true);
    if (resolvedLastFrame) payload.input_last_frame = formatMediaPayload(resolvedLastFrame);
    if (resolvedInputVideo) payload.input_video = formatMediaPayload(resolvedInputVideo);
  }

  // 云端任务若需要 input_image 但未解析出有效值，会在服务端报 'input_image'!；前端先校验并给出明确错误
  const tasksRequiringInputImage = ['i2v', 'i2i', 'animate', 's2v'];
  if (isCloud && tasksRequiringInputImage.includes(task) && (payload.input_image == null || payload.input_image === undefined)) {
    throw new Error(
      task === 'i2v' ? '请连接并确保图片输入节点已就绪（图片加载失败或未连接）' :
      task === 'i2i' ? '请连接并确保图片输入已就绪' :
      task === 'animate' ? '请连接图片与视频输入并确保已就绪' :
      task === 's2v' ? '请连接图片与音频输入并确保已就绪' :
      '缺少图片输入，请检查输入节点'
    );
  }

  // 对于 s2v 任务，如果使用多人模式，可能需要添加 negative_prompt
  // 主应用中 s2v 任务也会添加 negative_prompt（但内容略有不同）
  if (task === 's2v') {
    payload.negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
  }

  // 1. Submit Task (POST /api/v1/task/submit)
  // 日志：打印提交 payload 摘要（媒体字段只打 type 和 data 前 80 字符，避免 base64 刷屏）
  const summarizeMedia = (v: any) => {
    if (!v || typeof v !== 'object') return v;
    const d = v.data;
    const preview = typeof d === 'string' ? d.slice(0, 80) + (d.length > 80 ? '...' : '')
      : Array.isArray(d) ? d.map((x: any) => typeof x === 'string' ? x.slice(0, 80) + (x.length > 80 ? '...' : '') : x)
      : d;
    return { type: v.type, data: preview };
  };
  console.log('[LightX2V] Submit payload:', JSON.stringify({
    task: payload.task,
    model_cls: payload.model_cls,
    stage: payload.stage,
    seed: payload.seed,
    prompt: typeof payload.prompt === 'string' ? payload.prompt.slice(0, 100) : payload.prompt,
    aspect_ratio: payload.aspect_ratio,
    input_image: summarizeMedia(payload.input_image),
    input_audio: summarizeMedia(payload.input_audio),
    input_video: summarizeMedia(payload.input_video),
    input_last_frame: summarizeMedia(payload.input_last_frame),
  }, null, 2));


  let submitRes: Response;
  // retry 10 times if user visit too frequently
  for (let attempt = 0; attempt < 10; attempt++) {
    submitRes = await lightX2VRequest({
      baseUrl: actualBaseUrl,
      token: actualToken,
      endpoint: '/api/v1/task/submit',
      options: {
        method: 'POST',
        body: JSON.stringify(payload)
      }
    });
    if (submitRes.ok) { break; }
    if (submitRes.status === 400) {
      const errText = await submitRes.clone().text();
      if (errText.includes('visit too frequently')) {
        const wait_ms = Math.random() * 3000 + 500; // wait 500-3500ms
        console.warn(`[LightX2V] Submit rate limited, retry ${attempt + 1}/10 after ${wait_ms}ms`);
        await new Promise(r => setTimeout(r, wait_ms));
        continue;
      }
    }
    break;
  }

  if (!submitRes!.ok) {
    await handleLightX2VError(submitRes!, 'Submit');
  }

  const submitData = await submitRes.json();
  const taskId = submitData.task_id;
  if (!taskId) throw new Error("No task_id returned from LightX2V submission");

  // Call onTaskId callback if provided (for tracking task IDs for cancellation)
  if (onTaskId) {
    onTaskId(taskId);
  }

  // 2. Poll Task Status（与主应用一致：无 timeout，只要任务未失败/未取消就持续轮询）
  let status = "PENDING";

  while (status !== "SUCCEED" && status !== "FAILED" && status !== "CANCELLED") {
    if (abortSignal?.aborted) {
      throw new DOMException('Task cancelled by user', 'AbortError');
    }
    await new Promise(res => setTimeout(res, 5000));
    if (abortSignal?.aborted) {
      throw new DOMException('Task cancelled by user', 'AbortError');
    }
    const queryRes = await lightX2VRequest({
      baseUrl: actualBaseUrl,
      token: actualToken,
      endpoint: `/api/v1/task/query?task_id=${taskId}`,
      options: {
        method: 'GET'
      }
    });

    if (!queryRes.ok) {
      await handleLightX2VError(queryRes, `Polling (task_id: ${taskId})`);
    }
    const taskInfo = await queryRes.json();
    status = taskInfo.status;

    if (status === "FAILED") {
      throw new Error(`LightX2V Task Failed: ${taskInfo.error || 'Server processing error'}`);
    }
  }

  if (status !== "SUCCEED") {
    // 仅 CANCELLED 或其它非成功终态会走到这里（FAILED 已在上面 throw）
    throw new Error(status === "CANCELLED" ? 'LightX2V Task was cancelled' : `LightX2V Task ended with status: ${status}`);
  }

  const ref = toLightX2VResultRef(
    workflow_id,
    node_id,
    port_id,
    taskId,
    outputName,
    (modelCls || '').endsWith('-cloud')
  );
  return ref;
};

/**
 * LightX2V Result URL - resolve task_id + output_name to a fresh URL (for refs that avoid CDN expiry)
 */
export const lightX2VResultUrl = async (
  baseUrl: string,
  token: string,
  workflow_id: string,
  node_id: string,
  port_id: string,
  task_id: string,
): Promise<string> => {
  const url = `${baseUrl.replace(/\/$/, '')}/api/v1/workflow/${workflow_id}/node/${node_id}/output/${port_id}/url?task_id=${task_id}`;
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
  const res = await fetch(url, { method: 'GET', headers });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`LightX2V result_url failed: ${res.status} ${err}`);
  }
  const data = await res.json().catch(() => ({}));
  if (!data.url) throw new Error(`LightX2V result_url missing url for workflow_id=${workflow_id} node_id=${node_id} port_id=${port_id}`);
  return data.url;
};

/**
 * LightX2V Model List Service
 * Get available model list from LightX2V API
 */
export const lightX2VGetModelList = async (
  baseUrl: string,
  token: string
): Promise<Array<{ task: string; model_cls: string; stage: string }>> => {
  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint: '/api/v1/model/list',
    options: {
      method: 'GET'
    },
    requireToken: true // 使用 apiClient 时不需要 token
  });

  if (!response.ok) {
    await handleLightX2VError(response, 'Model List');
  }

  const data = await response.json();
  return data.models || [];
};

/**
 * LightX2V Cancel Task (for standalone: cancel via cloud API)
 */
export const lightX2VCancelTask = async (
  baseUrl: string,
  token: string,
  taskId: string
): Promise<Response> => {
  if (!baseUrl?.trim() || !taskId) {
    return new Response(JSON.stringify({ error: 'Missing baseUrl or taskId' }), { status: 400 });
  }
  const url = `${baseUrl.replace(/\/$/, '')}/api/v1/task/cancel?task_id=${encodeURIComponent(taskId)}`;
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
  return fetch(url, { method: 'GET', headers });
};

/**
 * LightX2V Task Query - get current task status (for post-cancel status sync)
 */
export const lightX2VTaskQuery = async (
  baseUrl: string,
  token: string,
  taskId: string
): Promise<{ status: string; error?: string }> => {
  if (!baseUrl?.trim() || !taskId) {
    return { status: 'UNKNOWN' };
  }
  const url = `${baseUrl.replace(/\/$/, '')}/api/v1/task/query?task_id=${encodeURIComponent(taskId)}`;
  const headers: Record<string, string> = {
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  };
  const res = await fetch(url, { method: 'GET', headers });
  if (!res.ok) return { status: 'UNKNOWN' };
  const data = await res.json().catch(() => ({}));
  return { status: data.status || 'UNKNOWN', error: data.error };
};

/**
 * Get LightX2V config for a specific model
 * Returns local or cloud config based on model ID
 */
export const getLightX2VConfigForModel = (modelId: string): { url: string; token: string; isCloud: boolean } => {
  const isCloud = modelId.endsWith('-cloud');

  if (isCloud) {
    // Cloud model - use cloud config
    const cloudUrl = (process.env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top').trim();
    const cloudToken = (process.env.LIGHTX2V_CLOUD_TOKEN || '').trim();

    if (!cloudToken) {
      throw new Error('LIGHTX2V_CLOUD_TOKEN 未设置。请设置 LIGHTX2V_CLOUD_TOKEN 环境变量以使用云端模型。');
    }

    return {
      url: cloudUrl,
      token: cloudToken,
      isCloud: true
    };
  } else {
    // Local model - use local config
    const envUrl = (process.env.LIGHTX2V_URL || '').trim();
    const apiClient = (window as any).__API_CLIENT__;

    // Get token from apiClient or environment
    let token: string;
    if (!envUrl && apiClient) {
      // Use apiClient (relative path) - token will be handled by apiClient
      token = '';
    } else {
      // Use direct URL - need token
      const localToken = localStorage.getItem('accessToken');
      token = localToken || (process.env.LIGHTX2V_TOKEN || '').trim();
    }

    return {
      url: envUrl || '',
      token: token,
      isCloud: false
    };
  }
};

/**
 * LightX2V TTS Voice List Service
 * Get available voice list from LightX2V TTS API
 */
export const lightX2VGetVoiceList = async (
  baseUrl: string,
  token: string,
  version: string = "all"
): Promise<{ voices?: any[]; emotions?: string[]; languages?: any[] }> => {
  const endpoint = `/api/v1/voices/list${version !== "all" ? `?version=${version}` : ''}`;
  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint,
    options: {
      method: 'GET'
    },
    requireToken: false // 使用 apiClient 时不需要 token
  });

  if (!response.ok) {
    await handleLightX2VError(response, 'Voice List');
  }

  let result: any;
  try {
    result = await response.json();
  } catch (jsonError: any) {
    const errorText = await response.text().catch(() => 'Unable to read response');
    console.error(`[LightX2V] Failed to parse JSON response:`, jsonError, errorText);
    throw new Error(`Failed to parse voice list response: ${jsonError.message || 'Invalid JSON'}`);
  }

  // Validate and normalize the response structure
  try {
    // Ensure voices is an array
    let voices: any[] = [];
    if (Array.isArray(result.voices)) {
      voices = result.voices;
    } else if (result.voices && typeof result.voices === 'object') {
      // If voices is an object, try to extract array from it
      console.warn('[LightX2V] Voices is not an array, attempting to normalize');
      voices = [];
    }

    // Ensure emotions is an array of strings
    let emotions: string[] = [];
    if (Array.isArray(result.emotions)) {
      // Check if emotions are objects with 'name' field or strings
      emotions = result.emotions.map((e: any) => {
        if (typeof e === 'string') return e;
        if (e && typeof e === 'object' && e.name) return e.name;
        return String(e);
      }).filter((e: any) => e);
    }

    // Ensure languages is an array
    let languages: any[] = [];
    if (Array.isArray(result.languages)) {
      languages = result.languages;
    }

    return {
      voices,
      emotions,
      languages
    };
  } catch (typeError: any) {
    console.error(`[LightX2V] Type error processing voice list:`, typeError, result);
    throw new Error(`Type error processing voice list: ${typeError.message || 'Invalid data structure'}`);
  }
};

/**
 * LightX2V TTS Service
 * Generate text-to-speech audio using LightX2V TTS API
 */
export const lightX2VTTS = async (
  baseUrl: string,
  token: string,
  text: string,
  voiceType: string,
  contextTexts: string = "",
  emotion: string = "",
  emotionScale: number = 3,
  speechRate: number = 0,
  pitch: number = 0,
  loudnessRate: number = 0,
  resourceId: string = "seed-tts-2.0"
): Promise<string> => {
  const textStr = typeof text === 'string' ? text : String(text || '');
  if (!textStr || !textStr.trim()) throw new Error("Text is required for TTS");
  const voiceTypeStr = typeof voiceType === 'string' ? voiceType : String(voiceType || '');
  if (!voiceTypeStr || !voiceTypeStr.trim()) throw new Error("Voice type is required for TTS");

  const payload = {
    text: textStr,
    voice_type: voiceTypeStr,
    context_texts: contextTexts,
    emotion: emotion,
    emotion_scale: emotionScale,
    speech_rate: speechRate,
    pitch: pitch,
    loudness_rate: loudnessRate,
    resource_id: resourceId
  };

  console.log('[TTS] Generate payload:', JSON.stringify({
    text: textStr.slice(0, 100) + (textStr.length > 100 ? '...' : ''),
    voice_type: voiceTypeStr,
    context_texts: contextTexts ? contextTexts.slice(0, 60) : '',
    emotion, emotion_scale: emotionScale,
    speech_rate: speechRate, pitch, loudness_rate: loudnessRate, resource_id: resourceId,
  }));

  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint: '/api/v1/tts/generate',
    options: {
    method: 'POST',
      body: JSON.stringify(payload),
      accept: 'application/json, audio/*'
    },
    requireToken: false // 使用 apiClient 时不需要 token
  });

  const contentType = response.headers.get("Content-Type") || "";

  if (!response.ok) {
    await handleLightX2VError(response, 'TTS');
  }

  // Check if response is audio or JSON error
  if (contentType.includes("audio") || contentType.includes("application/octet-stream")) {
    // Audio response - convert to base64 data URL
    const audioBlob = await response.blob();
    const audioBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        resolve(result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(audioBlob);
    });
    return audioBase64;
  } else {
    // Unexpected content type - try to read as text first, then parse if JSON
    try {
      const responseText = await response.text();
      // Try to parse as JSON
      if (responseText.trim().startsWith('{') || responseText.trim().startsWith('[')) {
        try {
          const errorData = JSON.parse(responseText);
          throw new Error(errorData.error || errorData.message || "TTS generation failed");
        } catch {
          // Not valid JSON, use text as error message
          throw new Error(responseText.trim() || "TTS generation failed");
        }
      } else {
        throw new Error(responseText.trim() || "TTS generation failed");
      }
    } catch (error: any) {
      // If already an Error with message, re-throw it
      if (error instanceof Error && error.message) {
        throw error;
      }
      // Otherwise create new error
      throw new Error("TTS generation failed: Unexpected response format");
    }
  }
};

/**
 * LightX2V Voice Clone Service
 * Clone voice from audio
 */
export const lightX2VVoiceClone = async (
  baseUrl: string,
  token: string,
  audioBase64: string,
  text?: string
): Promise<string> => {
  if (!audioBase64) throw new Error("Audio file is required for voice cloning");

  // Convert base64 to blob
  const base64Data = audioBase64.includes(',') ? audioBase64.split(',')[1] : audioBase64;
  const byteCharacters = atob(base64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: 'audio/wav' });

  const formData = new FormData();
  formData.append('file', blob, 'audio.wav');
  if (text) {
    formData.append('text', text);
  }

  console.log('[VoiceClone] Clone payload:', JSON.stringify({
    audio_size: byteArray.length,
    text: text ? text.slice(0, 100) : '(none)',
  }));

  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint: '/api/v1/voice/clone',
    options: {
    method: 'POST',
    body: formData
    },
    requireToken: false // 使用 apiClient 时不需要 token
  });

  if (!response.ok) {
    await handleLightX2VError(response, 'Voice Clone');
  }

  const result = await response.json();
  if (!result.speaker_id) {
    throw new Error(result.error || "Voice clone failed");
  }

  // Return speaker_id as JSON string (stored in node.data.speaker_id)
  return JSON.stringify({
    speaker_id: result.speaker_id,
    text: result.text || text || "",
    message: result.message || "Voice clone successful"
  });
};

/**
 * LightX2V Voice Clone TTS Service
 * Generate TTS with cloned voice
 */
export const lightX2VVoiceCloneTTS = async (
  baseUrl: string,
  token: string,
  text: string,
  speakerId: string,
  style: string = "正常",
  speed: number = 1.0,
  volume: number = 0,
  pitch: number = 0,
  language: string = "ZH_CN"
): Promise<string> => {
  const speakerIdStr = typeof speakerId === 'string' ? speakerId : String(speakerId || '');
  if (!speakerIdStr || !speakerIdStr.trim()) throw new Error("Speaker ID is required for TTS with cloned voice");
  const textStr = typeof text === 'string' ? text : String(text || '');
  if (!textStr || !textStr.trim()) throw new Error("Text is required for TTS");

  const payload = {
    text: textStr,
    speaker_id: speakerIdStr,
    style: style,
    speed: speed,
    volume: volume,
    pitch: pitch,
    language: language
  };

  console.log('[VoiceCloneTTS] Generate payload:', JSON.stringify({
    text: textStr.slice(0, 100) + (textStr.length > 100 ? '...' : ''),
    speaker_id: speakerIdStr,
    style, speed, volume, pitch, language,
  }));

  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint: '/api/v1/voice/clone/tts',
    options: {
    method: 'POST',
      body: JSON.stringify(payload),
      accept: 'application/json, audio/*'
    },
    requireToken: false // 使用 apiClient 时不需要 token
  });

  if (!response.ok) {
    await handleLightX2VError(response, 'Voice Clone TTS');
  }

  // Check if response is audio or JSON error
  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("audio") || contentType.includes("application/octet-stream")) {
    // Audio response - convert to base64 data URL
    const audioBlob = await response.blob();
    const audioBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        resolve(result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(audioBlob);
    });
    return audioBase64;
  } else {
    // JSON error response
    const errorData = await response.json();
    throw new Error(errorData.error || "TTS generation failed");
  }
};

/**
 * LightX2V Get Clone Voice List
 * Get list of cloned voices for the user
 */
export const lightX2VGetCloneVoiceList = async (
  baseUrl: string,
  token: string
): Promise<any[]> => {
  const response = await lightX2VRequest({
    baseUrl,
    token,
    endpoint: '/api/v1/voice/clone/list',
    options: {
      method: 'GET'
    },
    requireToken: false // 使用 apiClient 时不需要 token
  });

  if (!response.ok) {
    await handleLightX2VError(response, 'Clone Voice List');
  }

  const result = await response.json();
  try {
    let voices: any[] = [];

    // Check different possible response structures
    if (Array.isArray(result)) {
      // Direct array response
      voices = result;
    } else if (result.voice_clones && Array.isArray(result.voice_clones)) {
      // Response with voice_clones field
      voices = result.voice_clones;
    } else if (result.voices && Array.isArray(result.voices)) {
      // Response with voices field (fallback)
      voices = result.voices;
    } else if (result.data && Array.isArray(result.data)) {
      // Response with data field (fallback)
      voices = result.data;
    }

    console.log(`[LightX2V] Clone voice list parsed:`, {
      resultType: Array.isArray(result) ? 'array' : typeof result,
      hasVoiceClones: !!result.voice_clones,
      hasVoices: !!result.voices,
      voicesCount: voices.length,
      voices
    });

    return voices;
  } catch (typeError: any) {
    console.error(`[LightX2V] Type error processing clone voice list:`, typeError, result);
    throw new Error(`Type error processing clone voice list: ${typeError.message || 'Invalid data structure'}`);
  }
};

/**
 * DeepSeek Chat API Integration
 * Uses the DeepSeek API endpoint for responses (new API format)
 */
export const deepseekText = async (
  prompt: string,
  mode = 'basic',
  customInstruction?: string,
  model = 'deepseek-v3-2-251201',
  outputFields?: OutputField[],
  useSearch = false,
  returnRaw = false
): Promise<any> => {
  if (isStandalone()) {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey?.trim()) {
      throw new Error("API key is required. Please set DEEPSEEK_API_KEY environment variable.");
    }
  }

  // Ensure prompt is a string
  const promptStr = typeof prompt === 'string' ? prompt : String(prompt || '');

  const systemInstructions: Record<string, string> = {
    basic: "You are a helpful and versatile AI assistant. Provide clear, accurate, and direct answers.",
    enhance: "You are a Prompt Engineering Expert. Enhance the user's input into a detailed prompt. Output ONLY the enhanced prompt.",
    enhance_image: "Expand the user's input into a highly detailed image generation prompt. Output ONLY the prompt.",
    enhance_video: "Expand the user's input into a cinematic video prompt. Describe camera and lighting. Output ONLY the prompt.",
    enhance_tts: "Transform the input into a natural narration script. Output ONLY the text.",
    summarize: "Extract core info into a concise summary.",
    polish: "Refine text for clarity and tone.",
  };

  const baseInstruction = mode === 'custom' && customInstruction
    ? customInstruction
    : (systemInstructions[mode] || systemInstructions.basic);

  const hasMultipleOutputs = outputFields && outputFields.length > 0;
  const outputKeys = outputFields?.map(f => f.id) || [];

  // Build input array with user content
  const inputContent: any[] = [];

  // Add text content
  const textContent = hasMultipleOutputs
    ? `${promptStr}\n\nIMPORTANT: You MUST generate content for each field as JSON: ${outputKeys.join(', ')}.`
    : promptStr;

  inputContent.push({
    type: 'input_text',
    text: textContent
  });

  // Build request body for new responses API
  const requestBody: any = {
    model: model,
    stream: false,
    input: [
      {
        role: 'user',
        content: inputContent
      }
    ]
  };

  // Add system instruction if needed
  if (baseInstruction && mode !== 'basic') {
    requestBody.input.unshift({
      role: 'system',
      content: [
        {
          type: 'input_text',
          text: baseInstruction
        }
      ]
    });
  }

  // Add web search tool if useSearch is enabled
  if (useSearch) {
    requestBody.tools = [
      { type: "web_search" }
    ];
  }

  // Add JSON mode for structured output if multiple outputs are required
  // if (hasMultipleOutputs) {
  //   requestBody.response_format = { type: 'json_object' };
  // }

  const response = await volcResponsesRequest(requestBody);

  if (!response.ok) {
    let errorMessage = `DeepSeek API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  // Debug: log the full response structure for troubleshooting
  if (useSearch) {
    console.log('[DeepSeek] Full API response:', JSON.stringify(data, null, 2));
  }

  // New API format: response.output is an array, find the message type item
  // The message item has content array with type: "output_text"
  let text = "";
  if (data.output && Array.isArray(data.output)) {
    // Find all message type outputs (usually the last one contains the final answer)
    const messageOutputs = data.output.filter((item: any) => item.type === "message");
    // Use the last message output (final answer)
    const messageOutput = messageOutputs.length > 0 ? messageOutputs[messageOutputs.length - 1] : null;

    if (messageOutput && messageOutput.content && Array.isArray(messageOutput.content)) {
      // Find all output_text items and concatenate them
      const textContents = messageOutput.content.filter((item: any) => item.type === "output_text");
      if (textContents.length > 0) {
        // Concatenate all text contents
        text = textContents.map((item: any) => item.text || "").join("");
        if (useSearch) {
          console.log('[DeepSeek] Extracted text length:', text.length, 'characters');
        }
      }
    }

    // Debug log if text is empty
    if (!text) {
      console.warn('[DeepSeek] Failed to extract text from response:', JSON.stringify(data, null, 2));
    }
  }
  // Fallback to old format for backward compatibility
  if (!text) {
    text = data.output?.[0]?.content?.[0]?.text || data.choices?.[0]?.message?.content || "";
  }

  // Post-process: extract JSON from code blocks if present
  // The response API may return JSON wrapped in ```json ... ``` code blocks
  if (text.includes('```json')) {
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch && jsonMatch[1]) {
      try {
        const extractedJson = JSON.parse(jsonMatch[1].trim());
        // If hasMultipleOutputs is true, return the JSON object (already in correct format)
        if (hasMultipleOutputs) {
          outputKeys.forEach(key => { if (!(key in extractedJson)) extractedJson[key] = "..."; });
          return returnRaw
            ? { data: extractedJson, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
            : extractedJson;
        }
        // If hasMultipleOutputs is false but JSON contains output fields, extract the first value
        // This handles cases where API returns { "out-text": "..." } but we want just the text
        const jsonKeys = Object.keys(extractedJson);
        if (jsonKeys.length === 1 && jsonKeys[0].startsWith('out-')) {
          return extractedJson[jsonKeys[0]];
        }
        // If it matches expected output keys, extract the value
        if (outputKeys.length === 1 && outputKeys[0] in extractedJson) {
          return returnRaw
            ? { data: extractedJson[outputKeys[0]], raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
            : extractedJson[outputKeys[0]];
        }
        // Otherwise return the parsed JSON
        return returnRaw
          ? { data: extractedJson, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
          : extractedJson;
      } catch (e) {
        console.warn('[DeepSeek] Failed to parse JSON from code block:', e);
        // Fall through to normal processing
      }
    }
  }

  if (hasMultipleOutputs) {
    try {
      const parsed = JSON.parse(text);
      outputKeys.forEach(key => { if (!(key in parsed)) parsed[key] = "..."; });
      return returnRaw
        ? { data: parsed, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
        : parsed;
    } catch (e) {
      const fallback: Record<string, string> = {};
      outputKeys.forEach((key, i) => fallback[key] = i === 0 ? text : "...");
      return returnRaw
        ? { data: fallback, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
        : fallback;
    }
  }

  return returnRaw
    ? { data: text, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
    : text;
};

/**
 * Doubao Vision Chat API Integration
 * Uses the Doubao API endpoint for responses (new API format) with vision support
 */
export const doubaoText = async (
  prompt: string,
  mode = 'basic',
  customInstruction?: string,
  model = 'doubao-seed-1-6-vision-250815',
  outputFields?: OutputField[],
  imageInput?: string | string[] | any[],
  useSearch = false,
  returnRaw = false
): Promise<any> => {
  if (isStandalone()) {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey?.trim()) {
      throw new Error("API key is required. Please set DEEPSEEK_API_KEY environment variable.");
    }
  }

  // Ensure prompt is a string
  const promptStr = typeof prompt === 'string' ? prompt : String(prompt || '');

  const systemInstructions: Record<string, string> = {
    basic: "You are a helpful and versatile AI assistant. Provide clear, accurate, and direct answers.",
    enhance: "You are a Prompt Engineering Expert. Enhance the user's input into a detailed prompt. Output ONLY the enhanced prompt.",
    enhance_image: "Expand the user's input into a highly detailed image generation prompt. Output ONLY the prompt.",
    enhance_video: "Expand the user's input into a cinematic video prompt. Describe camera and lighting. Output ONLY the prompt.",
    enhance_tts: "Transform the input into a natural narration script. Output ONLY the text.",
    summarize: "Extract core info into a concise summary.",
    polish: "Refine text for clarity and tone.",
  };

  const baseInstruction = mode === 'custom' && customInstruction
    ? customInstruction
    : (systemInstructions[mode] || systemInstructions.basic);

  const hasMultipleOutputs = outputFields && outputFields.length > 0;
  const outputKeys = outputFields?.map(f => f.id) || [];

  // Build input content array - support both text and images
  const inputContent: any[] = [];

  // Add images if provided
  if (imageInput) {
    const images = Array.isArray(imageInput) ? imageInput : [imageInput];
    const flatImages = images.flat().filter(img => img && typeof img === 'string');
    for (const img of flatImages) {
      // Doubao API format: { "type": "input_image", "image_url": "https://..." or "data:image/..." }
      // For HTTP URLs, use directly; for base64/data URLs, use data URL format
      let imageUrl: string;

      if (img.startsWith('http://') || img.startsWith('https://')) {
        // HTTP/HTTPS URLs: use directly
        imageUrl = img;
      } else if (img.startsWith('data:')) {
        // Data URL: use directly (already in correct format)
        imageUrl = img;
      } else {
        // Base64 string without data URL prefix: convert to data URL
        // Try to detect mime type from common patterns
        let mimeType = 'image/jpeg';
        if (img.startsWith('/9j/') || img.startsWith('iVBORw0KGgo')) {
          mimeType = img.startsWith('/9j/') ? 'image/jpeg' : 'image/png';
        }
        imageUrl = `data:${mimeType};base64,${img}`;
      }

      inputContent.push({
        type: 'input_image',
        image_url: imageUrl
      });
    }
  }

  // Add text content
  const textContent = hasMultipleOutputs
    ? `${promptStr}\n\nIMPORTANT: You MUST generate content for each field as JSON: ${outputKeys.join(', ')}.`
    : promptStr;

  inputContent.push({
    type: 'input_text',
    text: textContent
  });

  // Build request body for new responses API
  const requestBody: any = {
    model: model,
    stream: false,
    input: [
      {
        role: 'user',
        content: inputContent
      }
    ]
  };

  // Add system instruction if needed
  if (baseInstruction && mode !== 'basic') {
    requestBody.input.unshift({
      role: 'system',
      content: [
        {
          type: 'input_text',
          text: baseInstruction
        }
      ]
    });
  }

  // Add web search tool if useSearch is enabled
  if (useSearch) {
    requestBody.tools = [
      { type: "web_search" }
    ];
  }

  // // Add JSON mode for structured output if multiple outputs are required
  // if (hasMultipleOutputs) {
  //   requestBody.response_format = { type: 'json_object' };
  // }

  const response = await volcResponsesRequest(requestBody);

  if (!response.ok) {
    let errorMessage = `Doubao API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  // Debug: log the full response structure for troubleshooting
  if (useSearch) {
    console.log('[Doubao] Full API response:', JSON.stringify(data, null, 2));
  }

  // New API format: response.output is an array, find the message type item
  // The message item has content array with type: "output_text"
  let text = "";
  if (data.output && Array.isArray(data.output)) {
    // Find all message type outputs (usually the last one contains the final answer)
    const messageOutputs = data.output.filter((item: any) => item.type === "message");
    // Use the last message output (final answer)
    const messageOutput = messageOutputs.length > 0 ? messageOutputs[messageOutputs.length - 1] : null;

    if (messageOutput && messageOutput.content && Array.isArray(messageOutput.content)) {
      // Find all output_text items and concatenate them
      const textContents = messageOutput.content.filter((item: any) => item.type === "output_text");
      if (textContents.length > 0) {
        // Concatenate all text contents
        text = textContents.map((item: any) => item.text || "").join("");
        if (useSearch) {
          console.log('[Doubao] Extracted text length:', text.length, 'characters');
        }
      }
    }

    // Debug log if text is empty
    if (!text) {
      console.warn('[Doubao] Failed to extract text from response:', JSON.stringify(data, null, 2));
    }
  }
  // Fallback to old format for backward compatibility
  if (!text) {
    text = data.output?.[0]?.content?.[0]?.text || data.choices?.[0]?.message?.content || "";
  }

  // Post-process: extract JSON from code blocks if present
  // The response API may return JSON wrapped in ```json ... ``` code blocks
  if (text.includes('```json')) {
    const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
    if (jsonMatch && jsonMatch[1]) {
      try {
        const extractedJson = JSON.parse(jsonMatch[1].trim());
        // If hasMultipleOutputs is true, return the JSON object (already in correct format)
        if (hasMultipleOutputs) {
          outputKeys.forEach(key => { if (!(key in extractedJson)) extractedJson[key] = "..."; });
          return returnRaw
            ? { data: extractedJson, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
            : extractedJson;
        }
        // If hasMultipleOutputs is false but JSON contains output fields, extract the first value
        // This handles cases where API returns { "out-text": "..." } but we want just the text
        const jsonKeys = Object.keys(extractedJson);
        if (jsonKeys.length === 1 && jsonKeys[0].startsWith('out-')) {
          return extractedJson[jsonKeys[0]];
        }
        // If it matches expected output keys, extract the value
        if (outputKeys.length === 1 && outputKeys[0] in extractedJson) {
          return returnRaw
            ? { data: extractedJson[outputKeys[0]], raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
            : extractedJson[outputKeys[0]];
        }
        // Otherwise return the parsed JSON
        return returnRaw
          ? { data: extractedJson, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
          : extractedJson;
      } catch (e) {
        console.warn('[Doubao] Failed to parse JSON from code block:', e);
        // Fall through to normal processing
      }
    }
  }

  if (hasMultipleOutputs) {
    try {
      const parsed = JSON.parse(text);
      outputKeys.forEach(key => { if (!(key in parsed)) parsed[key] = "..."; });
      return returnRaw
        ? { data: parsed, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
        : parsed;
    } catch (e) {
      const fallback: Record<string, string> = {};
      outputKeys.forEach((key, i) => fallback[key] = i === 0 ? text : "...");
      return returnRaw
        ? { data: fallback, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
        : fallback;
    }
  }

  return returnRaw
    ? { data: text, raw_response: data, usage: data.usage || {}, finish_reason: data.finish_reason || '' }
    : text;
};

/**
 * PP Chat Gemini API Integration
 * Uses the PP Chat API endpoint for chat completions (custom Gemini endpoint)
 * Supports both text and image inputs
 */
export const ppchatGeminiText = async (
  prompt: string,
  mode = 'basic',
  customInstruction?: string,
  model = 'gemini-3-pro-preview',
  outputFields?: OutputField[],
  imageInput?: string | string[] | any[]
): Promise<any> => {
  const apiKey = process.env.PPCHAT_API_KEY;
  if (!apiKey) {
    throw new Error("PP Chat API key is required. Please set PPCHAT_API_KEY environment variable.");
  }

  // Ensure prompt is a string
  const promptStr = typeof prompt === 'string' ? prompt : String(prompt || '');

  const systemInstructions: Record<string, string> = {
    basic: "You are a helpful and versatile AI assistant. Provide clear, accurate, and direct answers.",
    enhance: "You are a Prompt Engineering Expert. Enhance the user's input into a detailed prompt. Output ONLY the enhanced prompt.",
    enhance_image: "Expand the user's input into a highly detailed image generation prompt. Output ONLY the prompt.",
    enhance_video: "Expand the user's input into a cinematic video prompt. Describe camera and lighting. Output ONLY the prompt.",
    enhance_tts: "Transform the input into a natural narration script. Output ONLY the text.",
    summarize: "Extract core info into a concise summary.",
    polish: "Refine text for clarity and tone.",
  };

  const baseInstruction = mode === 'custom' && customInstruction
    ? customInstruction
    : (systemInstructions[mode] || systemInstructions.basic);

  const hasMultipleOutputs = outputFields && outputFields.length > 0;
  const outputKeys = outputFields?.map(f => f.id) || [];

  // Build parts array - support both text and images
  const parts: any[] = [];

  // Add images if provided
  if (imageInput) {
    const images = Array.isArray(imageInput) ? imageInput : [imageInput];
    const flatImages = images.flat().filter(img => img && typeof img === 'string');
    flatImages.forEach(img => {
      // Extract base64 data and mime type from data URL or base64 string
      let base64Data: string;
      let mimeType: string = 'image/jpeg';

      if (img.startsWith('data:')) {
        // Data URL format: data:image/jpeg;base64,/9j/4AAQ...
        const matches = img.match(/^data:([^;]+);base64,(.+)$/);
        if (matches) {
          mimeType = matches[1] || 'image/jpeg';
          base64Data = matches[2];
        } else {
          // Fallback: try to extract base64 from data URL without explicit mime type
          const base64Match = img.match(/base64,(.+)$/);
          base64Data = base64Match ? base64Match[1] : img;
        }
      } else if (img.startsWith('http')) {
        // If it's a URL, we need to fetch it first (for now, skip URLs)
        // In a production environment, you might want to fetch and convert
        return;
      } else {
        // Assume it's already base64
        base64Data = img;
      }

      parts.push({
        inline_data: {
          mime_type: mimeType,
          data: base64Data
        }
      });
    });
  }

  // Add text content
  const textContent = hasMultipleOutputs
    ? `${promptStr}\n\nIMPORTANT: You MUST generate content for each field as JSON: ${outputKeys.join(', ')}.`
    : promptStr;

  if (textContent) {
    parts.push({ text: textContent });
  }

  // Build request body
  const requestBody: any = {
    contents: [{
      parts: parts
    }]
  };

  // Add system instruction if needed
  if (baseInstruction && mode !== 'basic') {
    requestBody.systemInstruction = {
      parts: [{ text: baseInstruction }]
    };
  }

  // Add instruction for multiple outputs if needed
  if (hasMultipleOutputs) {
    requestBody.generationConfig = {
      responseMimeType: "application/json",
      responseSchema: {
        type: "object",
        properties: (outputFields || []).reduce((acc, field) => ({
          ...acc,
          [field.id]: { type: "string", description: field.description || field.id }
        }), {}),
        required: outputKeys
      }
    };
  }

  const response = await fetch(`https://api.ppchat.vip/v1beta/models/${model}:generateContent`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-goog-api-key': apiKey
    },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    let errorMessage = `PP Chat API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

  if (hasMultipleOutputs) {
    try {
      const parsed = JSON.parse(text);
      outputKeys.forEach(key => { if (!(key in parsed)) parsed[key] = "..."; });
      return parsed;
    } catch (e) {
      const fallback: Record<string, string> = {};
      outputKeys.forEach((key, i) => fallback[key] = i === 0 ? text : "...");
      return fallback;
    }
  }

  return text;
};

/**
 * PP Chat OpenAI-format Chat Completions API
 * Uses the /v1/chat/completions endpoint with OpenAI-compatible format
 */
export const ppchatChatCompletions = async (
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  model = 'gemini-3-pro-preview',
  responseFormat: 'json_object' | 'text' = 'json_object'
): Promise<string> => {
  const apiKey = process.env.PPCHAT_API_KEY;
  if (!apiKey) {
    throw new Error("PP Chat API key is required. Please set PPCHAT_API_KEY environment variable.");
  }

  // 根据 OpenAI 格式，messages 数组应该支持 system、user、assistant 角色
  // 直接使用 messages，但确保格式正确
  const formattedMessages = messages.map(msg => ({
    role: msg.role,
    content: msg.content
  }));

  const requestBody: any = {
    model: model,
    stream: false,
    messages: formattedMessages
  };

  // 添加 JSON 模式支持
  if (responseFormat === 'json_object') {
    requestBody.response_format = { type: 'json_object' };
  }

  // 构建 Cookie header（如果需要）
  const cookieValue = process.env.PPCHAT_COOKIE || '';
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  };

  // 如果有 Cookie，添加到 headers
  if (cookieValue) {
    headers['Cookie'] = cookieValue;
  }

  const response = await fetch('https://api.ppchat.vip/v1/chat/completions', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    let errorMessage = `PP Chat API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();

  // 提取 content 从 OpenAI 格式的响应
  const content = data.choices?.[0]?.message?.content || '';

  if (!content) {
    throw new Error('Empty response from PP Chat API');
  }

  return content;
};

/**
 * DeepSeek Chat Completions API for structured JSON output
 * Uses the /api/v3/chat/completions endpoint with JSON mode
 */
export const deepseekChat = async (
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  model = 'deepseek-v3-2-251201',
  responseFormat: 'json_object' | 'text' = 'json_object'
): Promise<string> => {
  if (isStandalone()) {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey?.trim()) {
      throw new Error("DeepSeek API key is required. Please set DEEPSEEK_API_KEY environment variable.");
    }
  }

  console.log('deepseekChat messages:', messages);

  // 转换为 Responses API 格式
  // 根据文档：
  // - 当使用 text 参数时，input 中的 content 应该是字符串
  // - 当不使用 text 参数时，input 中的 content 应该是数组 [{ type: "input_text", text: "..." }]
  const inputList: any[] = [];
  const useTextParam = responseFormat === 'json_object';

  for (const msg of messages) {
    const role = msg.role;
    const content = msg.content;

    // 跳过空内容的消息
    if (!content || (typeof content === 'string' && content.trim() === '') ||
        (Array.isArray(content) && content.length === 0)) {
      continue;
    }

    if (useTextParam) {
      // 使用 text 参数时，content 直接是字符串
      let textContent: string;
      if (typeof content === 'string') {
        textContent = content.trim();
      } else if (Array.isArray(content)) {
        // 从数组中提取文本内容（忽略图片等其他类型）
        const textItems = (content as any[]).filter((item: any) => item.type === 'input_text');
        textContent = textItems.map((item: any) => item.text || '').join(' ').trim();
        // 如果没有文本内容，跳过这条消息
        if (!textContent) {
          continue;
        }
      } else {
        textContent = String(content).trim();
      }

      // 确保 content 不为空
      if (!textContent) {
        continue;
      }

      inputList.push({
        role: role,
        content: textContent
      });
    } else {
      // 不使用 text 参数时，使用数组格式
      let contentList: any[];
      if (typeof content === 'string') {
        const trimmedContent = content.trim();
        if (!trimmedContent) {
          continue;
        }
        contentList = [{ type: 'input_text', text: trimmedContent }];
      } else if (Array.isArray(content)) {
        // 过滤掉空内容
        contentList = (content as any[]).filter((item: any) => {
          if (item.type === 'input_text') {
            return item.text && item.text.trim();
          }
          return true; // 保留非文本类型（如图片）
        });
        if (contentList.length === 0) {
          continue;
        }
      } else {
        const strContent = String(content).trim();
        if (!strContent) {
          continue;
        }
        contentList = [{ type: 'input_text', text: strContent }];
      }

      inputList.push({
        role: role,
        content: contentList
      });
    }
  }

  // 确保至少有一条消息
  if (inputList.length === 0) {
    throw new Error('No valid messages to send');
  }

  const requestBody: any = {
    model: model,
    stream: false
  };

  // 根据文档，如果只有一条 user 消息且没有 system 消息，可以直接使用字符串
  // 否则使用数组格式
  if (inputList.length === 1 && inputList[0].role === 'user' && useTextParam) {
    requestBody.input = inputList[0].content; // 直接使用字符串
  } else {
    requestBody.input = inputList; // 使用数组格式
  }

  // 结构化输出通过顶层的 text 参数传递
  if (useTextParam) {
    requestBody.text = { format: { type: 'json_object' } };
  }

  // 调试日志
  console.log('[DeepSeek Chat] Request body:', JSON.stringify(requestBody, null, 2));

  const response = await volcResponsesRequest(requestBody);

  if (!response.ok) {
    let errorMessage = `DeepSeek Responses API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();
  console.log('deepseekChat data:', data);

  // Extract content from Responses API format
  let text = '';
  if (data.output && Array.isArray(data.output)) {
    // 查找所有 message 类型的输出
    const messageOutputs = data.output.filter((item: any) => item.type === 'message');

    // 使用最后一个 message 输出（通常是最终答案）
    if (messageOutputs.length > 0) {
      const messageOutput = messageOutputs[messageOutputs.length - 1];
      const content = messageOutput.content || [];

      if (Array.isArray(content)) {
        // 查找所有 output_text 类型的内容
        const textContents = content.filter((item: any) => item.type === 'output_text');
        if (textContents.length > 0) {
          // 拼接所有文本内容
          text = textContents.map((item: any) => item.text || '').join('');
        }
      }
    }
  }

  // 向后兼容：尝试旧格式
  if (!text) {
    text = data.output?.[0]?.content?.[0]?.text || data.choices?.[0]?.message?.content || '';
  }

  if (!text) {
    throw new Error('Empty response from DeepSeek Responses API');
  }

  return text;
};

/**
 * DeepSeek Chat Completions API with streaming support
 * Returns an async generator that yields chunks of the response
 */
export async function* deepseekChatStream(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  model = 'deepseek-v3-2-251201',
  responseFormat: 'json_object' | 'text' = 'json_object',
  signal?: AbortSignal
): AsyncGenerator<{ type: 'thinking' | 'content'; text: string }, void, unknown> {
  if (isStandalone()) {
    const apiKey = process.env.DEEPSEEK_API_KEY;
    if (!apiKey?.trim()) {
      throw new Error("DeepSeek API key is required. Please set DEEPSEEK_API_KEY environment variable.");
    }
  }

  // 转换为 Responses API 格式
  // 根据文档：
  // - 当使用 text 参数时，input 中的 content 应该是字符串
  // - 当不使用 text 参数时，input 中的 content 应该是数组 [{ type: "input_text", text: "..." }]
  const inputList: any[] = [];
  const useTextParam = responseFormat === 'json_object';

  for (const msg of messages) {
    const role = msg.role;
    const content = msg.content;

    // 跳过空内容的消息
    if (!content || (typeof content === 'string' && content.trim() === '') ||
        (Array.isArray(content) && content.length === 0)) {
      continue;
    }

    if (useTextParam) {
      // 使用 text 参数时，content 直接是字符串
      let textContent: string;
      if (typeof content === 'string') {
        textContent = content.trim();
      } else if (Array.isArray(content)) {
        // 从数组中提取文本内容（忽略图片等其他类型）
        const textItems = (content as any[]).filter((item: any) => item.type === 'input_text');
        textContent = textItems.map((item: any) => item.text || '').join(' ').trim();
        // 如果没有文本内容，跳过这条消息
        if (!textContent) {
          continue;
        }
      } else {
        textContent = String(content).trim();
      }

      // 确保 content 不为空
      if (!textContent) {
        continue;
      }

      inputList.push({
        role: role,
        content: textContent
      });
    } else {
      // 不使用 text 参数时，使用数组格式
      let contentList: any[];
      if (typeof content === 'string') {
        const trimmedContent = content.trim();
        if (!trimmedContent) {
          continue;
        }
        contentList = [{ type: 'input_text', text: trimmedContent }];
      } else if (Array.isArray(content)) {
        // 过滤掉空内容
        contentList = (content as any[]).filter((item: any) => {
          if (item.type === 'input_text') {
            return item.text && item.text.trim();
          }
          return true; // 保留非文本类型（如图片）
        });
        if (contentList.length === 0) {
          continue;
        }
      } else {
        const strContent = String(content).trim();
        if (!strContent) {
          continue;
        }
        contentList = [{ type: 'input_text', text: strContent }];
      }

      inputList.push({
        role: role,
        content: contentList
      });
    }
  }

  // 确保至少有一条消息
  if (inputList.length === 0) {
    throw new Error('No valid messages to send');
  }

  const requestBody: any = {
    model: model,
    stream: true,
    thinking: { type: 'enabled' } // 启用思考过程
  };

  // 根据文档，如果只有一条 user 消息且没有 system 消息，可以直接使用字符串
  // 否则使用数组格式
  if (inputList.length === 1 && inputList[0].role === 'user' && useTextParam) {
    requestBody.input = inputList[0].content; // 直接使用字符串
  } else {
    requestBody.input = inputList; // 使用数组格式
  }

  // 结构化输出通过顶层的 text 参数传递
  if (useTextParam) {
    requestBody.text = { format: { type: 'json_object' } };
  }

  // 调试日志
  console.log('[DeepSeek Chat Stream] Request body:', JSON.stringify(requestBody, null, 2));

  const response = await volcResponsesRequest(requestBody, signal);

  if (!response.ok) {
    let errorMessage = `DeepSeek Responses API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('Failed to get response reader');
  }

  let buffer = '';
  let currentEvent: string | null = null;

  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmedLine = line.trim();

        if (trimmedLine === '') {
          // 空行表示一个事件结束，重置 currentEvent
          currentEvent = null;
          continue;
        }

        // 解析 SSE 格式：event: xxx 和 data: {...}
        if (trimmedLine.startsWith('event: ')) {
          currentEvent = trimmedLine.slice(7).trim();
          continue;
        }

        if (trimmedLine.startsWith('data: ')) {
          const dataStr = trimmedLine.slice(6);
          if (dataStr === '[DONE]') {
            return;
          }

          try {
            const data = JSON.parse(dataStr);

            // 根据事件类型处理（优先使用 currentEvent，如果没有则使用 data.type）
            const eventType = currentEvent || data.type;

            // 处理思考过程的增量文本
            if (eventType === 'response.reasoning_summary_text.delta') {
              if (data.delta) {
                yield { type: 'thinking', text: data.delta };
              }
            }
            // 处理消息内容的增量文本
            else if (eventType === 'response.message_text.delta') {
              if (data.delta) {
                yield { type: 'content', text: data.delta };
              }
            }
            // 处理其他可能包含文本的事件
            else if (data.delta && typeof data.delta === 'string') {
              // 如果有 delta 字段，尝试根据上下文判断类型
              if (eventType?.includes('reasoning') || eventType?.includes('thinking')) {
                yield { type: 'thinking', text: data.delta };
              } else {
                yield { type: 'content', text: data.delta };
              }
            }
            // 向后兼容：尝试解析 output 数组格式
            else if (data.output && Array.isArray(data.output)) {
              for (const outputItem of data.output) {
                if (outputItem.type === 'reasoning' || outputItem.type === 'thinking') {
                  const content = outputItem.content || [];
                  if (Array.isArray(content)) {
                    const textContents = content.filter((item: any) => item.type === 'output_text');
                    for (const textItem of textContents) {
                      if (textItem.text) {
                        yield { type: 'thinking', text: textItem.text };
                      }
                    }
                  }
                } else if (outputItem.type === 'message') {
                  const content = outputItem.content || [];
                  if (Array.isArray(content)) {
                    const textContents = content.filter((item: any) => item.type === 'output_text');
                    for (const textItem of textContents) {
                      if (textItem.text) {
                        yield { type: 'content', text: textItem.text };
                      }
                    }
                  }
                }
              }
            }
            // 向后兼容：尝试旧格式（chat/completions 格式）
            else {
              const choice = data.choices?.[0];
              const delta = choice?.delta;

              if (delta) {
                const isThinkingChunk = choice?.type === 'thinking' || delta.type === 'thinking';

                if (isThinkingChunk) {
                  const thinkingContent = delta.content || delta.text || delta.thinking || '';
                  if (thinkingContent) {
                    yield { type: 'thinking', text: thinkingContent };
                  }
                } else if (delta.thinking !== undefined && delta.thinking !== null && delta.thinking !== '') {
                  yield { type: 'thinking', text: delta.thinking };
                }

                if (!isThinkingChunk) {
                  const content = delta.content || '';
                  if (content) {
                    yield { type: 'content', text: content };
                  }
                }
              }
            }
          } catch (e) {
            // Skip invalid JSON
            console.warn('Failed to parse SSE data:', dataStr, e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * PP Chat Completions API with streaming support
 */
export async function* ppchatChatCompletionsStream(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  model = 'gemini-3-pro-preview',
  responseFormat: 'json_object' | 'text' = 'json_object',
  signal?: AbortSignal
): AsyncGenerator<{ type: 'thinking' | 'content'; text: string }, void, unknown> {
  const apiKey = process.env.PPCHAT_API_KEY;
  if (!apiKey) {
    throw new Error("PP Chat API key is required. Please set PPCHAT_API_KEY environment variable.");
  }

  const formattedMessages = messages.map(msg => ({
    role: msg.role,
    content: msg.content
  }));

  const requestBody: any = {
    model: model,
    stream: true,
    messages: formattedMessages
  };

  if (responseFormat === 'json_object') {
    requestBody.response_format = { type: 'json_object' };
  }

  const cookieValue = process.env.PPCHAT_COOKIE || '';
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${apiKey}`
  };

  if (cookieValue) {
    headers['Cookie'] = cookieValue;
  }

  const response = await fetch('https://api.ppchat.vip/v1/chat/completions', {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(requestBody),
    signal
  });

  if (!response.ok) {
    let errorMessage = `PP Chat API failed (${response.status})`;
    try {
      const errorData = await response.json();
      errorMessage = errorData.error?.message || errorData.error || errorMessage;
    } catch (e) {
      const errorText = await response.text();
      errorMessage = errorText || errorMessage;
    }
    throw new Error(errorMessage);
  }

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('Failed to get response reader');
  }

  let buffer = '';

  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim() === '') continue;
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          if (dataStr === '[DONE]') {
            return;
          }

          try {
            const data = JSON.parse(dataStr);
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const content = delta.content || '';
              if (content) {
                yield { type: 'content', text: content };
              }
            }
          } catch (e) {
            console.warn('Failed to parse SSE data:', dataStr);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
