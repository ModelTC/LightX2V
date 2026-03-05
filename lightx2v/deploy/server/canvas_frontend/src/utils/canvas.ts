// 画布工具函数

export interface ViewState {
  x: number;
  y: number;
  zoom: number;
}

/**
 * 屏幕坐标转世界坐标
 */
export const screenToWorld = (
  screenX: number,
  screenY: number,
  view: ViewState,
  canvasRect?: DOMRect
): { x: number; y: number } => {
  return {
    x: (screenX - view.x) / view.zoom,
    y: (screenY - view.y) / view.zoom
  };
};

/**
 * 世界坐标转屏幕坐标
 */
export const worldToScreen = (
  worldX: number,
  worldY: number,
  view: ViewState
): { x: number; y: number } => {
  return {
    x: worldX * view.zoom + view.x,
    y: worldY * view.zoom + view.y
  };
};

/**
 * 计算连接路径
 */
export const calculateConnectionPath = (
  x1: number,
  y1: number,
  x2: number,
  y2: number
): string => {
  return `M ${x1} ${y1} C ${x1 + 100} ${y1}, ${x2 - 100} ${y2}, ${x2} ${y2}`;
};
