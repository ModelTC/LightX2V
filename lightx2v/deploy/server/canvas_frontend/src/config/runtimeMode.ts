/**
 * 运行模式：纯前端部署（仅用 Cloud 接口，不接自建后端）vs 有后端
 * 通过环境变量 VITE_STANDALONE 控制，构建时注入
 */
export function isStandalone(): boolean {
  const v = typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_STANDALONE;
  return v === true || v === 'true' || v === '1';
}
