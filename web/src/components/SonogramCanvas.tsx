import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ToolMode, LayerVisibilityState, ViewportMode } from '../types/ultrasound';
import { Maximize2, ZoomIn, ZoomOut, Eye, Layers } from 'lucide-react';

interface SonogramCanvasProps {
  viewportMode: ViewportMode;
  onSetViewportMode: (mode: ViewportMode) => void;
  currentFrameImage: string | null;
  currentFrameIdx: number;
  totalFrames: number;
  superficialMm: number[];
  deepMm: number[];
  redMm: number[];
  timestamps: number[];
  toolMode: ToolMode;
  brushRadius: number;
  zoom: number;
  onSetZoom: (z: number) => void;
  layerVisibility: LayerVisibilityState;
  pixelSpacingMm: number;
  maxDepthMm: number;
  onRedCurveChange: (newRedMm: number[]) => void;
  onHoverData?: (timeSec: number, frame: number, depthMm: number, thicknessMm: number) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement>;
  currentMask?: Uint8Array | null;
}

export const SonogramCanvas: React.FC<SonogramCanvasProps> = ({
  viewportMode,
  onSetViewportMode,
  currentFrameImage,
  currentFrameIdx,
  totalFrames,
  superficialMm,
  deepMm,
  redMm,
  timestamps,
  toolMode,
  brushRadius,
  zoom,
  onSetZoom,
  layerVisibility,
  pixelSpacingMm,
  maxDepthMm,
  onRedCurveChange,
  onHoverData,
  canvasRef: externalCanvasRef,
  currentMask,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const internalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const isDrawingRef = useRef(false);
  const imgElementRef = useRef<HTMLImageElement | null>(null);

  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({
    width: 800,
    height: 600,
  });
  const [imgNaturalSize, setImgNaturalSize] = useState<{ width: number; height: number }>({
    width: 219,
    height: 512,
  });

  // Track parent container dimensions via ResizeObserver with immediate mount probe
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setContainerSize({
          width: Math.floor(rect.width),
          height: Math.floor(rect.height),
        });
      }
    };

    updateSize();

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setContainerSize({ width: Math.floor(width), height: Math.floor(height) });
        }
      }
    });

    observer.observe(el);
    window.addEventListener('resize', updateSize);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', updateSize);
    };
  }, []);

  // Load and decode image
  useEffect(() => {
    if (!currentFrameImage) {
      imgElementRef.current = null;
      renderCanvas();
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = currentFrameImage;
    img.onload = () => {
      imgElementRef.current = img;
      setImgNaturalSize({
        width: img.naturalWidth || 219,
        height: img.naturalHeight || 512,
      });
      renderCanvas();
    };
  }, [currentFrameImage]);

  // Compute responsive canvas dimensions with NO aspect ratio clipping
  const isSideBySide = viewportMode === 'side_by_side';
  const imgW = imgNaturalSize.width;
  const imgH = imgNaturalSize.height;

  // Margin padding inside canvas container (preserve clearance for top HUD toolbar)
  const padX = 24;
  const padY = 56;
  const availW = Math.max(120, containerSize.width - padX);
  const availH = Math.max(120, containerSize.height - padY);

  let canvasW = 600;
  let canvasH = 500;

  if (viewportMode === 'filmstrip') {
    // Panoramic filmstrip view
    if (zoom <= 1.0) {
      canvasW = availW;
      canvasH = Math.min(availH, Math.max(300, Math.round(availW * 0.35)));
    } else {
      canvasW = Math.round(availW * zoom);
      canvasH = Math.min(availH, 480);
    }
  } else {
    // Cine modes: overlay, side_by_side, raw
    const aspect = isSideBySide ? (imgW * 2) / imgH : imgW / imgH;

    if (zoom <= 1.0) {
      // Fit to Screen (100% completely visible inside container without cut off)
      if (availW / availH > aspect) {
        canvasH = availH;
        canvasW = Math.round(availH * aspect);
      } else {
        canvasW = availW;
        canvasH = Math.round(availW / aspect);
      }
    } else {
      // Zoomed in
      canvasH = Math.round(availH * zoom);
      canvasW = Math.round(canvasH * aspect);
    }
  }

  // Calculate actual total depth in mm of this ultrasound image
  // For 512 px height with 0.1757 mm/px calibration => total depth is 89.96 mm
  const effectiveTotalDepthMm = Math.max(
    10,
    maxDepthMm > 0 && maxDepthMm < 150 ? maxDepthMm : imgH * pixelSpacingMm
  );

  // Generate offscreen mask canvas whenever currentMask or opacity changes
  const maskCanvas = React.useMemo(() => {
    if (!currentMask) return null;
    const w = imgNaturalSize.width;
    const h = imgNaturalSize.height;
    if (w <= 0 || h <= 0) return null;

    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return null;

    const imgData = offCtx.createImageData(w, h);
    const data = imgData.data;
    const alphaVal = Math.round(255 * layerVisibility.maskOpacity * 0.45);

    const len = Math.min(currentMask.length, w * h);
    for (let i = 0; i < len; i++) {
      if (currentMask[i] > 0) {
        const idx = i * 4;
        data[idx] = 129;     // Soft Sage Green (matching Python [129, 145, 126])
        data[idx + 1] = 145;
        data[idx + 2] = 126;
        data[idx + 3] = alphaVal;
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    return offscreen;
  }, [currentMask, layerVisibility.maskOpacity, imgNaturalSize.width, imgNaturalSize.height]);

  // Main Render Routine
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasW, canvasH);

    // Dark neutral artboard background
    ctx.fillStyle = '#0a0a0c';
    ctx.fillRect(0, 0, canvasW, canvasH);

    const fIdx = Math.min(Math.max(0, currentFrameIdx), totalFrames - 1);
    const supDepth = superficialMm[fIdx] || 14.0;
    const deepDepth = deepMm[fIdx] || 28.0;

    // Convert depth in mm to canvas pixel Y
    const depthToY = (depthMm: number) => {
      return (depthMm / effectiveTotalDepthMm) * canvasH;
    };

    if (viewportMode === 'side_by_side') {
      // -------------------------------------------------------------
      // 1. SIDE-BY-SIDE DUAL VIEW (LEFT: RAW, RIGHT: AI MASK)
      // -------------------------------------------------------------
      const halfW = canvasW / 2;

      // Left Panel: Raw B-Mode
      if (imgElementRef.current) {
        ctx.drawImage(imgElementRef.current, 0, 0, halfW, canvasH);
      } else {
        ctx.fillStyle = '#111215';
        ctx.fillRect(0, 0, halfW, canvasH);
      }

      // Right Panel: Raw B-Mode + AI Overlays
      if (imgElementRef.current) {
        ctx.drawImage(imgElementRef.current, halfW, 0, halfW, canvasH);
      } else {
        ctx.fillStyle = '#111215';
        ctx.fillRect(halfW, 0, halfW, canvasH);
      }

      const ySup = depthToY(supDepth);
      const yDeep = depthToY(deepDepth);

      // Mask overlay on right panel
      if (layerVisibility.showMask) {
        if (maskCanvas) {
          ctx.drawImage(maskCanvas, halfW, 0, halfW, canvasH);
        } else {
          ctx.save();
          ctx.fillStyle = `rgba(129, 145, 126, ${layerVisibility.maskOpacity * 0.45})`;
          ctx.fillRect(halfW, ySup, halfW, Math.max(0, yDeep - ySup));
          ctx.restore();
        }
      }

      // Superficial aponeurosis on right panel
      if (layerVisibility.showSuperficial) {
        ctx.strokeStyle = '#a8c7fa';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(halfW, ySup);
        ctx.lineTo(canvasW, ySup);
        ctx.stroke();

        ctx.fillStyle = '#a8c7fa';
        ctx.font = 'bold 10px Segoe UI, sans-serif';
        ctx.fillText(`SUP: ${supDepth.toFixed(1)}mm`, halfW + 6, Math.max(14, ySup - 4));
      }

      // Deep aponeurosis on right panel
      if (layerVisibility.showDeep) {
        ctx.strokeStyle = '#d0bcff';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(halfW, yDeep);
        ctx.lineTo(canvasW, yDeep);
        ctx.stroke();

        ctx.fillStyle = '#d0bcff';
        ctx.font = 'bold 10px Segoe UI, sans-serif';
        ctx.fillText(`DEEP: ${deepDepth.toFixed(1)}mm`, halfW + 6, Math.min(canvasH - 6, yDeep + 12));
      }

      // Center divider line
      ctx.strokeStyle = '#37393b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(halfW, 0);
      ctx.lineTo(halfW, canvasH);
      ctx.stroke();

      // Badges
      ctx.fillStyle = 'rgba(19, 19, 20, 0.85)';
      ctx.fillRect(6, 6, 85, 18);
      ctx.fillStyle = '#c4c7c5';
      ctx.font = 'bold 9px monospace';
      ctx.fillText('RAW B-MODE', 10, 18);

      ctx.fillStyle = 'rgba(19, 19, 20, 0.85)';
      ctx.fillRect(halfW + 6, 6, 100, 18);
      ctx.fillStyle = '#a8c7fa';
      ctx.fillText('AI SEGMENTATION', halfW + 10, 18);

    } else if (viewportMode === 'overlay' || viewportMode === 'raw') {
      // -------------------------------------------------------------
      // 2. OVERLAY OR RAW B-MODE CINE VIEW
      // -------------------------------------------------------------
      if (imgElementRef.current) {
        ctx.drawImage(imgElementRef.current, 0, 0, canvasW, canvasH);
      } else {
        ctx.fillStyle = '#111215';
        ctx.fillRect(0, 0, canvasW, canvasH);
      }

      if (viewportMode === 'overlay') {
        const ySup = depthToY(supDepth);
        const yDeep = depthToY(deepDepth);

        // A. Biceps muscle mask overlay
        if (layerVisibility.showMask) {
          if (maskCanvas) {
            ctx.drawImage(maskCanvas, 0, 0, canvasW, canvasH);
          } else {
            ctx.save();
            ctx.fillStyle = `rgba(129, 145, 126, ${layerVisibility.maskOpacity * 0.45})`;
            ctx.fillRect(0, ySup, canvasW, Math.max(0, yDeep - ySup));
            ctx.restore();
          }
        }

        // B. Superficial fascia line (Gemini Blue)
        if (layerVisibility.showSuperficial) {
          ctx.strokeStyle = '#a8c7fa';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(0, ySup);
          ctx.lineTo(canvasW, ySup);
          ctx.stroke();

          ctx.fillStyle = '#a8c7fa';
          ctx.font = 'bold 11px Segoe UI, sans-serif';
          ctx.fillText(`SUP: ${supDepth.toFixed(1)}mm`, 8, Math.max(14, ySup - 4));
        }

        // C. Deep fascia line (Gemini Purple)
        if (layerVisibility.showDeep) {
          ctx.strokeStyle = '#d0bcff';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(0, yDeep);
          ctx.lineTo(canvasW, yDeep);
          ctx.stroke();

          ctx.fillStyle = '#d0bcff';
          ctx.font = 'bold 11px Segoe UI, sans-serif';
          ctx.fillText(`DEEP: ${deepDepth.toFixed(1)}mm`, 8, Math.min(canvasH - 6, yDeep + 14));
        }
      }

    } else {
      // -------------------------------------------------------------
      // 3. PANORAMIC FILMSTRIP VIEWPORT
      // -------------------------------------------------------------
      if (imgElementRef.current) {
        ctx.drawImage(imgElementRef.current, 0, 0, canvasW, canvasH);
      } else {
        ctx.fillStyle = '#111215';
        ctx.fillRect(0, 0, canvasW, canvasH);
      }

      const n = timestamps.length;
      if (n >= 2) {
        // Muscle mask shaded zone
        if (layerVisibility.showMask) {
          ctx.save();
          ctx.fillStyle = `rgba(196, 238, 208, ${layerVisibility.maskOpacity * 0.4})`;
          ctx.beginPath();
          for (let i = 0; i < n; i++) {
            const x = (i / (n - 1)) * canvasW;
            const y = depthToY(superficialMm[i] || 14.0);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          for (let i = n - 1; i >= 0; i--) {
            const x = (i / (n - 1)) * canvasW;
            const y = depthToY(deepMm[i] || 28.0);
            ctx.lineTo(x, y);
          }
          ctx.closePath();
          ctx.fill();
          ctx.restore();
        }

        // Superficial Line
        if (layerVisibility.showSuperficial) {
          ctx.strokeStyle = '#a8c7fa';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          for (let i = 0; i < n; i++) {
            const x = (i / (n - 1)) * canvasW;
            const y = depthToY(superficialMm[i] || 14.0);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        // Deep Fascia Line
        if (layerVisibility.showDeep) {
          ctx.strokeStyle = '#d0bcff';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          for (let i = 0; i < n; i++) {
            const x = (i / (n - 1)) * canvasW;
            const y = depthToY(deepMm[i] || 28.0);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        // Interactive Red Line
        ctx.strokeStyle = '#f28b82';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = (i / (n - 1)) * canvasW;
          const y = depthToY(redMm[i] || 32.0);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Playback Needle
        const needleX = (currentFrameIdx / Math.max(1, totalFrames - 1)) * canvasW;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(needleX, 0);
        ctx.lineTo(needleX, canvasH);
        ctx.stroke();
      }
    }

    // Depth Grid (10mm to 80mm based on effectiveTotalDepthMm)
    if (layerVisibility.showGrid) {
      ctx.font = 'bold 10px Segoe UI, sans-serif';
      for (let depth = 10; depth <= Math.floor(effectiveTotalDepthMm); depth += 10) {
        const y = depthToY(depth);
        if (y < canvasH - 4) {
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.28)';
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvasW, y);
          ctx.stroke();
          ctx.setLineDash([]);

          // Depth Label Chip
          ctx.fillStyle = 'rgba(26, 27, 30, 0.85)';
          ctx.fillRect(4, y - 8, 40, 16);
          ctx.fillStyle = '#c4c7c5';
          ctx.fillText(`${depth}mm`, 6, y + 4);
        }
      }
    }

    // Brush cursor preview
    if (mousePos && (toolMode === 'pen' || toolMode === 'eraser')) {
      ctx.save();
      ctx.beginPath();
      ctx.arc(mousePos.x, mousePos.y, brushRadius, 0, Math.PI * 2);
      ctx.strokeStyle = toolMode === 'pen' ? '#f28b82' : '#a8c7fa';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }
  }, [
    viewportMode,
    currentFrameIdx,
    totalFrames,
    superficialMm,
    deepMm,
    redMm,
    timestamps,
    layerVisibility,
    canvasW,
    canvasH,
    mousePos,
    toolMode,
    brushRadius,
    effectiveTotalDepthMm,
  ]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Mouse event handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (toolMode !== 'pen' && toolMode !== 'eraser') return;
    isDrawingRef.current = true;
    applyBrush(e);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    // Hover telemetry calculation
    if (onHoverData && timestamps.length > 0) {
      const ratio = Math.max(0, Math.min(1, x / canvasW));
      const frameIdx = Math.round(ratio * (timestamps.length - 1));
      const tSec = timestamps[frameIdx] || 0;
      const depth = (y / canvasH) * effectiveTotalDepthMm;
      const th = Math.abs((deepMm[frameIdx] || 28) - (superficialMm[frameIdx] || 14));
      onHoverData(tSec, frameIdx, depth, th);
    }

    if (isDrawingRef.current && (toolMode === 'pen' || toolMode === 'eraser')) {
      applyBrush(e);
    }
  };

  const handleMouseUp = () => {
    isDrawingRef.current = false;
  };

  const handleMouseLeave = () => {
    isDrawingRef.current = false;
    setMousePos(null);
  };

  const applyBrush = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const n = timestamps.length;
    if (n < 2) return;

    const targetDepthMm = (y / canvasH) * effectiveTotalDepthMm;
    const centerIdx = Math.round((x / canvasW) * (n - 1));
    const spreadFrames = Math.max(1, Math.round((brushRadius / canvasW) * n * 1.5));

    const updated = [...redMm];
    for (let i = Math.max(0, centerIdx - spreadFrames); i <= Math.min(n - 1, centerIdx + spreadFrames); i++) {
      const dist = Math.abs(i - centerIdx) / Math.max(1, spreadFrames);
      const weight = Math.cos(dist * (Math.PI / 2));

      if (toolMode === 'pen') {
        updated[i] = updated[i] * (1 - weight) + targetDepthMm * weight;
      } else if (toolMode === 'eraser') {
        const resetDepth = (deepMm[i] || 28.0) + 6.0;
        updated[i] = updated[i] * (1 - weight) + resetDepth * weight;
      }
    }
    onRedCurveChange(updated);
  };

  return (
    <div
      ref={containerRef}
      className="flex-1 flex flex-col items-center justify-center bg-[var(--canvas-bg)] pt-14 pb-3 px-3 overflow-auto relative select-none min-h-0 min-w-0"
    >
      {/* Top Floating Viewport Control Bar */}
      <div className="absolute top-4 left-6 z-20 flex items-center gap-1.5 bg-[var(--bg-surface)]/90 backdrop-blur-md px-2.5 py-1.5 rounded-xl border border-[var(--border-subtle)] shadow-xl text-xs font-semibold">
        {/* View Mode Pills (Overlay, Side-by-Side, Raw, Filmstrip) */}
        <button
          onClick={() => onSetViewportMode('overlay')}
          className={`px-2 py-1 rounded-md transition ${
            viewportMode === 'overlay'
              ? 'bg-[var(--gemini-blue)] text-black font-bold shadow-sm'
              : 'text-[var(--text-secondary)] hover:bg-[var(--bg-card)]'
          }`}
        >
          Soft Overlay
        </button>
        <button
          onClick={() => onSetViewportMode('side_by_side')}
          className={`px-2 py-1 rounded-md transition ${
            viewportMode === 'side_by_side'
              ? 'bg-[var(--gemini-blue)] text-black font-bold shadow-sm'
              : 'text-[var(--text-secondary)] hover:bg-[var(--bg-card)]'
          }`}
        >
          Side-by-Side Dual
        </button>
        <button
          onClick={() => onSetViewportMode('raw')}
          className={`px-2 py-1 rounded-md transition ${
            viewportMode === 'raw'
              ? 'bg-[var(--gemini-blue)] text-black font-bold shadow-sm'
              : 'text-[var(--text-secondary)] hover:bg-[var(--bg-card)]'
          }`}
        >
          Raw B-Mode
        </button>
        <button
          onClick={() => onSetViewportMode('filmstrip')}
          className={`px-2 py-1 rounded-md transition ${
            viewportMode === 'filmstrip'
              ? 'bg-[var(--gemini-blue)] text-black font-bold shadow-sm'
              : 'text-[var(--text-secondary)] hover:bg-[var(--bg-card)]'
          }`}
        >
          Filmstrip
        </button>

        <div className="w-[1px] h-4 bg-[var(--border-subtle)] mx-1" />

        {/* Zoom Controls */}
        <button
          onClick={() => onSetZoom(1.0)}
          className="px-2 py-1 rounded-md text-[var(--text-secondary)] hover:bg-[var(--bg-card)] hover:text-[var(--text-primary)] transition flex items-center gap-1"
          title="Fit Screen (Ctrl+0)"
        >
          <Maximize2 size={12} />
          <span>Fit Screen</span>
        </button>
        <button
          onClick={() => onSetZoom(Math.max(0.5, Number((zoom - 0.25).toFixed(2))))}
          className="p-1 rounded-md text-[var(--text-secondary)] hover:bg-[var(--bg-card)] hover:text-[var(--text-primary)] transition"
          title="Zoom Out (-)"
        >
          <ZoomOut size={13} />
        </button>
        <span className="font-mono text-[11px] text-[var(--text-muted)] w-10 text-center">
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={() => onSetZoom(Math.min(2.5, Number((zoom + 0.25).toFixed(2))))}
          className="p-1 rounded-md text-[var(--text-secondary)] hover:bg-[var(--bg-card)] hover:text-[var(--text-primary)] transition"
          title="Zoom In (+)"
        >
          <ZoomIn size={13} />
        </button>
      </div>

      {/* Photoshop Document Canvas (Centered with Drop Shadow) */}
      <div
        className="rounded-xl overflow-hidden border border-[var(--border-subtle)] shadow-2xl relative bg-black transition-all"
        style={{ width: `${canvasW}px`, height: `${canvasH}px` }}
      >
        <canvas
          ref={canvasRef}
          width={canvasW}
          height={canvasH}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          className={`block w-full h-full ${
            toolMode === 'pen' || toolMode === 'eraser'
              ? 'cursor-crosshair'
              : toolMode === 'pan'
              ? 'cursor-grab'
              : 'cursor-default'
          }`}
        />
      </div>
    </div>
  );
};
