/**
 * 格式化时间
 */
export const formatTime = (ms?: number): string => {
  if (ms === undefined) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
};

/** 运行时间：与主应用一致，显示为「X分Y秒」 */
export const formatRunTime = (ms?: number): string => {
  if (ms === undefined || ms < 0) return '';
  const totalSec = Math.floor(ms / 1000);
  const mins = Math.floor(totalSec / 60);
  const secs = totalSec % 60;
  if (mins > 0) return `${mins}分${secs}秒`;
  return `${secs}秒`;
};
