import { DataType } from '../../types';
import { getAccessToken, getApiBaseUrl } from './apiClient';

/**
 * 判断是否为需要带鉴权请求的 URL（同源或相对路径），
 * 此类 URL 若直接用 <a> 打开会拿到 HTML 错误页而非文件
 */
function isAuthRequiredUrl(content: string): boolean {
  if (typeof content !== 'string' || !content) return false;
  if (content.startsWith('data:') || content.startsWith('blob:')) return false;
  if (content.startsWith('/')) return true;
  if (content.startsWith('http://') || content.startsWith('https://')) {
    try {
      return new URL(content).origin === window.location.origin;
    } catch {
      return false;
    }
  }
  return false;
}

/**
 * 获取用于 fetch 的完整 URL
 */
function getFetchUrl(content: string): string {
  if (content.startsWith('http://') || content.startsWith('https://')) return content;
  const base = getApiBaseUrl();
  return base ? `${base.replace(/\/$/, '')}${content.startsWith('/') ? '' : '/'}${content}` : content;
}

/** 确保文件名带扩展名，否则浏览器可能不触发保存或保存为未知类型 */
function ensureExtension(fileName: string, type: DataType): string {
  const trimmed = (fileName || 'download').trim();
  if (/\.(png|jpe?g|gif|webp|mp4|webm|mov|wav|mp3|ogg|m4a)$/i.test(trimmed)) return trimmed;
  switch (type) {
    case DataType.IMAGE: return `${trimmed}.png`;
    case DataType.VIDEO: return `${trimmed}.mp4`;
    case DataType.AUDIO: return `${trimmed}.wav`;
    default: return trimmed;
  }
}

/**
 * 下载文件
 * - 文本：生成 Blob 后下载
 * - 图片/视频/音频：若为同源或相对路径则带鉴权 fetch 成 Blob 再下载，避免下到 HTML 错误页；否则直接使用 URL
 */
export async function downloadFile(
  content: string,
  fileName: string,
  type: DataType
): Promise<void> {
  const name = ensureExtension(fileName, type);
  const link = document.createElement('a');

  if (type === DataType.TEXT) {
    const contentString = typeof content === 'object'
      ? JSON.stringify(content, null, 2)
      : content;
    const blob = new Blob([contentString], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    return;
  }

  // 图片/视频/音频
  if (isAuthRequiredUrl(content)) {
    const token = getAccessToken();
    const fullUrl = getFetchUrl(content);
    const res = await fetch(fullUrl, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) {
      console.error('[download] Fetch failed:', res.status, fullUrl);
      throw new Error(`下载失败: ${res.status}`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    link.href = url;
    link.download = name;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    // 延迟撤销，否则部分浏览器尚未开始保存就失效导致下载失败
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    return;
  }

  link.href = content;
  link.download = name;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
