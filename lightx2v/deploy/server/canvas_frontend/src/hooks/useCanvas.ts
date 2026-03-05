import React, { useState, useCallback, useRef } from 'react';
import { ViewState, screenToWorld } from '../utils/canvas';
import { WorkflowState } from '../../types';

export const useCanvas = (workflow: WorkflowState | null, canvasRef: React.RefObject<HTMLDivElement>) => {
  const [view, setView] = useState<ViewState>({ x: 0, y: 0, zoom: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isOverNode, setIsOverNode] = useState(false);
  const [draggingNode, setDraggingNode] = useState<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const [connecting, setConnecting] = useState<{
    nodeId: string;
    portId: string;
    type: any;
    direction: 'in' | 'out';
    startX: number;
    startY: number;
  } | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [clientMousePos, setClientMousePos] = useState({ clientX: 0, clientY: 0 });

  // 当画布在父级 scale(0.8) 等缩放容器内时，getBoundingClientRect() 是渲染尺寸，需转为逻辑像素
  const getContainerScale = useCallback((): number => {
    const el = canvasRef.current;
    const rect = el?.getBoundingClientRect();
    if (!el || !rect || rect.width === 0) return 1;
    const scale = rect.width / (el.offsetWidth || rect.width);
    return Number.isFinite(scale) && scale > 0 ? scale : 1;
  }, [canvasRef]);

  // Convert screen coordinates to world coordinates (x,y 为相对画布的像素，会先按 containerScale 转为逻辑像素)
  const screenToWorldCoords = useCallback((x: number, y: number) => {
    const scale = getContainerScale();
    const logicalX = x / scale;
    const logicalY = y / scale;
    return screenToWorld(logicalX, logicalY, view, canvasRef.current?.getBoundingClientRect());
  }, [view, canvasRef, getContainerScale]);

  // Zoom in
  const zoomIn = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const newZoom = Math.min(view.zoom * 1.2, 5);
      const zoomRatio = newZoom / view.zoom;
      setView({
        zoom: newZoom,
        x: centerX - (centerX - view.x) * zoomRatio,
        y: centerY - (centerY - view.y) * zoomRatio
      });
    }
  }, [view, canvasRef]);

  // Zoom out
  const zoomOut = useCallback(() => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const newZoom = Math.max(view.zoom / 1.2, 0.1);
      const zoomRatio = newZoom / view.zoom;
      setView({
        zoom: newZoom,
        x: centerX - (centerX - view.x) * zoomRatio,
        y: centerY - (centerY - view.y) * zoomRatio
      });
    }
  }, [view, canvasRef]);

  // Reset view to center nodes
  const resetView = useCallback(() => {
    if (workflow && workflow.nodes.length > 0) {
      const nodes = workflow.nodes;
      const avgX = nodes.reduce((sum, n) => sum + n.x, 0) / nodes.length;
      const avgY = nodes.reduce((sum, n) => sum + n.y, 0) / nodes.length;
      setView({ x: -avgX + 400, y: -avgY + 300, zoom: 1 });
    } else {
      setView({ x: 0, y: 0, zoom: 1 });
    }
  }, [workflow]);

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const scale = getContainerScale();
    const viewX = (e.clientX - rect.left) / scale;
    const viewY = (e.clientY - rect.top) / scale;
    setMousePos({ x: viewX, y: viewY });
    setClientMousePos({ clientX: e.clientX, clientY: e.clientY });

    if (isPanning) {
      const deltaX = viewX - panStart.x;
      const deltaY = viewY - panStart.y;
      setView(prev => ({
        ...prev,
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      setPanStart({ x: viewX, y: viewY });
    }

    if (draggingNode) {
      const world = screenToWorldCoords(viewX, viewY);
      // Node dragging is handled by parent component
    }
  }, [isPanning, panStart, draggingNode, canvasRef, screenToWorldCoords, getContainerScale]);

  // Handle mouse down
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.node-element, .port, button, input, textarea, label')) {
      return;
    }

    if (e.button === 0) { // Left mouse button
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const scale = getContainerScale();
      const viewX = (e.clientX - rect.left) / scale;
      const viewY = (e.clientY - rect.top) / scale;
      setIsPanning(true);
      setPanStart({ x: viewX, y: viewY });
    }
  }, [canvasRef, getContainerScale]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    setDraggingNode(null);
    setConnecting(null);
    setIsPanning(false);
  }, []);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setIsOverNode(false);
  }, []);

  // Handle wheel (zoom)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    // Detect zoom gesture:
    // 1. Ctrl/Cmd + wheel (desktop zoom)
    // 2. Trackpad pinch (ctrlKey + deltaY on macOS)
    // 3. Trackpad zoom gestures (deltaY with small deltaX, or when deltaY is much larger)
    const isTrackpadPinch = e.ctrlKey || e.metaKey;
    const isTrackpadZoom = !isTrackpadPinch && Math.abs(e.deltaY) > 0 && (Math.abs(e.deltaX) < 5 || Math.abs(e.deltaY) / Math.abs(e.deltaX) > 2);
    const isZoom = isTrackpadPinch || isTrackpadZoom;

    if (isZoom) {
      e.preventDefault();

      // Get mouse position relative to canvas (逻辑像素，与 view 一致)
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const scale = getContainerScale();
      const mouseX = (e.clientX - rect.left) / scale;
      const mouseY = (e.clientY - rect.top) / scale;

      // Calculate zoom factor - adjust sensitivity for trackpad vs mouse
      let zoomFactor: number;
      if (e.deltaMode === 0) {
        // Pixel scrolling (trackpad or high-resolution mouse)
        // Use smaller factor for smoother zoom
        zoomFactor = 1 - (e.deltaY * 0.0008);
      } else {
        // Line/page scrolling (traditional mouse wheel)
        zoomFactor = 1 - (e.deltaY * 0.01);
      }

      const newZoom = Math.min(Math.max(view.zoom * zoomFactor, 0.1), 5);

      // Zoom towards mouse position (keep the point under cursor fixed)
      const zoomRatio = newZoom / view.zoom;
      setView(v => ({
        zoom: newZoom,
        x: mouseX - (mouseX - v.x) * zoomRatio,
        y: mouseY - (mouseY - v.y) * zoomRatio
      }));
    } else {
      // Pan with trackpad/wheel (when not zooming)，delta 按容器缩放修正
      e.preventDefault();
      const scale = getContainerScale();
      setView(v => ({ ...v, x: v.x - e.deltaX / scale, y: v.y - e.deltaY / scale }));
    }
  }, [view, canvasRef, getContainerScale]);

  // Node drag handlers
  const handleNodeDragStart = useCallback((nodeId: string, offsetX: number, offsetY: number) => {
    setDraggingNode({ id: nodeId, offsetX, offsetY });
  }, []);

  const handleNodeDrag = useCallback((nodeId: string, x: number, y: number) => {
    if (!draggingNode || draggingNode.id !== nodeId) return;
    // Node position update is handled by parent component
  }, [draggingNode]);

  const handleNodeDragEnd = useCallback(() => {
    setDraggingNode(null);
  }, []);

  return {
    view,
    setView,
    isPanning,
    isOverNode,
    setIsOverNode,
    draggingNode,
    connecting,
    setConnecting,
    mousePos,
    clientMousePos,
    zoomIn,
    zoomOut,
    resetView,
    handleMouseMove,
    handleMouseDown,
    handleMouseUp,
    handleMouseLeave,
    handleWheel,
    handleNodeDragStart,
    handleNodeDrag,
    handleNodeDragEnd,
    screenToWorldCoords
  };
};
