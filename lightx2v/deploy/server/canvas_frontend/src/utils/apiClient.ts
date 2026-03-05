import { isStandalone } from '../config/runtimeMode';

/**
 * 初始化 LIGHTX2V_TOKEN
 * - 现在为「本地后端」或「Cloud」两种模式：LIGHTX2V_TOKEN 可选
 * - 若设置了 LIGHTX2V_URL：优先用环境变量 LIGHTX2V_TOKEN，否则用用户登录的 accessToken
 * - 若未设置 LIGHTX2V_URL：用用户登录的 accessToken 或环境变量 LIGHTX2V_TOKEN
 */
export function initLightX2VToken(): void {
  const envUrl = (process.env.BASE_URL || '').trim();
  const envToken = (process.env.LIGHTX2V_TOKEN || '').trim();
  const accessToken = localStorage.getItem('accessToken');

  if (envUrl && envToken) {
    // 环境变量 URL + TOKEN 都有：直接使用
    console.log('[LightX2V] 使用环境变量 URL + LIGHTX2V_TOKEN');
    return;
  }
  if (accessToken) {
    (process.env as any).LIGHTX2V_TOKEN = accessToken;
    console.log('[LightX2V] 使用用户登录的 accessToken 作为 LIGHTX2V_TOKEN');
    return;
  }
  if (envToken) {
    (process.env as any).LIGHTX2V_TOKEN = envToken;
    console.log('[LightX2V] 使用环境变量 LIGHTX2V_TOKEN');
    return;
  }
  // 本地后端或未配置：不设置 token，由请求时 getAccessToken 再取
  console.log('[LightX2V] 未设置 LIGHTX2V_TOKEN，将使用本地后端或登录后 token');
}

/**
 * 获取 API 客户端实例
 */
export function getApiClient(): any {
  return (window as any).__API_CLIENT__ || null;
}

/**
 * 获取共享存储实例
 */
export function getSharedStore(): any {
  return (window as any).__SHARED_STORE__ || null;
}

/**
 * 统一的 API 请求函数
 * 优先使用主应用的 apiClient，否则使用直接 fetch
 */
/**
 * 获取认证 headers（参考主应用的 getAuthHeaders）
 */
function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };

  // 优先从 sharedStore 获取 token，否则从 localStorage 获取
  const sharedStore = getSharedStore();
  const token = sharedStore ? sharedStore.getState('token') : localStorage.getItem('accessToken');

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
    console.log('[apiRequest] 使用Token进行认证:', token.substring(0, 20) + '...');
  } else {
    console.warn('[apiRequest] 没有找到accessToken');
  }

  return headers;
}

export async function apiRequest(url: string, options: RequestInit = {}): Promise<Response> {
  // 纯前端部署时，不请求自建后端（相对路径均为后端）；LightX2V Cloud 通过 lightX2VRequest 打完整 URL
  if (isStandalone() && !url.startsWith('http://') && !url.startsWith('https://')) {
    throw new Error('[Standalone] Backend API disabled. Use local storage or LightX2V Cloud only.');
  }

  const apiClient = getApiClient();

  // 参考主应用的做法：总是获取认证 headers，然后合并 options.headers
  // options.headers 会覆盖 getAuthHeaders() 中的 headers
  const authHeaders = getAuthHeaders();
  const incomingHeaders = (options.headers as Record<string, string>) || {};

  // 合并 headers：先使用 authHeaders，然后 options.headers 会覆盖
  const headers: Record<string, string> = {
    ...authHeaders,
    ...incomingHeaders
  };

  // 如果有 apiClient，使用它
  if (apiClient && typeof apiClient.request === 'function') {
    // apiClient 使用相对路径，但需要完整 URL
    // 获取 baseURL
    const baseURL = apiClient.baseURL || '';
    const fullUrl = url.startsWith('http') ? url : `${baseURL}${url}`;

    // 使用 apiClient 的 request 方法
    // 但 apiClient.request 返回的是解析后的 JSON，我们需要 Response 对象
    // 所以需要直接使用 fetch，但使用 apiClient 的配置
    return fetch(fullUrl, {
      ...options,
      headers
    });
  }

  // 回退到直接 fetch（使用相对路径，浏览器会自动使用当前域名）
  return fetch(url, {
    ...options,
    headers
  });
}

/**
 * 获取后端基础 URL
 */
export function getApiBaseUrl(): string {
  const apiClient = getApiClient();
  if (apiClient && apiClient.baseURL) {
    return apiClient.baseURL;
  }
  // 回退到当前域名
  return typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.host}`
    : 'http://localhost:8081';
}

/**
 * 获取访问令牌
 * - 本地后端或 Cloud：优先 sharedStore（qiankun 主应用），否则 localStorage，否则环境变量 LIGHTX2V_TOKEN
 */
let tokenCache: { token: string; timestamp: number } | null = null;
const TOKEN_CACHE_TTL = 1000;

/** 清除 token 缓存，用于 qiankun mount 后强制重新从 sharedStore/localStorage 读取 */
export function clearTokenCache(): void {
  tokenCache = null;
}

export function getAccessToken(): string {
  const now = Date.now();
  if (tokenCache && now - tokenCache.timestamp < TOKEN_CACHE_TTL) {
    return tokenCache.token;
  }

  const sharedStore = getSharedStore();
  const storeToken = sharedStore ? sharedStore.getState('token') : null;
  const localToken = typeof localStorage !== 'undefined' ? localStorage.getItem('accessToken') : null;
  const envToken = (process.env.LIGHTX2V_TOKEN || '').trim();

  const token = (storeToken || localToken || envToken) || '';
  if (storeToken || localToken) {
    (process.env as any).LIGHTX2V_TOKEN = token;
  }
  tokenCache = { token, timestamp: now };
  return token;
}
