import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ToolMode } from '../types/ultrasound';
import { mmToPixel, pixelToMm, frameToPixelX, pixelXToFrame } from '../utils/calibration';

interface SonogramCanvasProps {
  imageSrc: string | null;
  superficialMm: number[];
  deepMm: number[];
  redMm: number[];
  timestamps: number[];
  toolMode: ToolMode;
  brushRadius: number;
  zoom: number;
  onRedCurveChange: (newRedMm: number[]) => void;
  onHoverData?: (timeSec: number, frame: number, depthMm: number, thicknessMm: number) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement>;
}

export const SonogramCanvas: React.FC<SonogramCanvasProps> = ({
  imageSrc,
  superficialMm,
  deepMm,
  redMm,
  timestamps,
  toolMode,
  brushRadius,
  zoom,
  onRedCurveChange,
  onHoverData,
  canvasRef: externalCanvasRef,
}) => {
  const internalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = externalCanvasRef || internalCanvasRef;
  const isDrawingRef = useRef(false);
  const imgElementRef = useRef<HTMLImageElement | null>(null);

  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);

  const baseWidth = 2400;
  const canvasHeight = 520;
  const canvasWidth = Math.round(baseWidth * zoom);
  const maxDepthMm = 89.96;

  // Load and cache background image
  useEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageSrc;
    img.onload = () => {
      imgElementRef.current = img;
      renderCanvas();
    };
  }, [imageSrc]);

  // Main Render Loop
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 1. Draw Background Ultrasound Sonogram
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    if (imgElementRef.current) {
      ctx.drawImage(imgElementRef.current, 0, 0, canvasWidth, canvasHeight);
    } else {
      // Procedural fallback medical grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvasWidth; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvasHeight);
        ctx.stroke();
      }
    }

    const totalFrames = timestamps.length;
    if (totalFrames < 2) return;

    // 2. Depth Reference Grid (Dashed lines at 10, 20, 30, 40, 50, 60, 70, 80 mm)
    ctx.font = 'bold 11px Segoe UI, sans-serif';
    for (let depth = 10; depth <= 80; depth += 10) {
      const y = mmToPixel(depth, canvasHeight, maxDepthMm);
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.25)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvasWidth, y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Depth Badge
      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(8, y - 9, 44, 18);
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(`${depth} mm`, 12, y + 4);
    }

    // 3. Time Reference Grid (Every 5 or 10 seconds)
    const totalTime = timestamps[totalFrames - 1] || 1;
    const timeStep = totalTime <= 40 ? 5 : 10;
    for (let t = timeStep; t < totalTime; t += timeStep) {
      // Find frame closest to time t
      let closestFrame = 0;
      let minDiff = 9999;
      for (let i = 0; i < totalFrames; i++) {
        const diff = Math.abs(timestamps[i] - t);
        if (diff < minDiff) {
          minDiff = diff;
          closestFrame = i;
        }
      }
      const x = frameToPixelX(closestFrame, totalFrames, canvasWidth);

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvasHeight);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
      ctx.fillRect(x - 16, canvasHeight - 20, 32, 16);
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`${t}s`, x - 10, canvasHeight - 8);
    }

    // 4. Muscle Compartment Shaded Fill (Orange polygon between Yellow and Red curves)
    ctx.fillStyle = 'rgba(249, 115, 22, 0.35)';
    ctx.beginPath();
    // Top boundary (yellow) forward
    for (let i = 0; i < totalFrames; i++) {
      const x = frameToPixelX(i, totalFrames, canvasWidth);
      const y = mmToPixel(deepMm[i] ?? 24.0, canvasHeight, maxDepthMm);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    // Bottom boundary (red) backward
    for (let i = totalFrames - 1; i >= 0; i--) {
      const x = frameToPixelX(i, totalFrames, canvasWidth);
      const y = mmToPixel(redMm[i] ?? (deepMm[i] ?? 24.0) + 12.0, canvasHeight, maxDepthMm);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();

    // 5. Superficial Fascia (Cyan #00f2fe)
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i < totalFrames; i++) {
      const x = frameToPixelX(i, totalFrames, canvasWidth);
      const y = mmToPixel(superficialMm[i] ?? 12.0, canvasHeight, maxDepthMm);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 6. Deep Fascia (Yellow #fde047)
    ctx.strokeStyle = '#fde047';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i < totalFrames; i++) {
      const x = frameToPixelX(i, totalFrames, canvasWidth);
      const y = mmToPixel(deepMm[i] ?? 24.0, canvasHeight, maxDepthMm);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 7. Interactive Kinetic Red Line (#ff4444)
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 3.0;
    ctx.beginPath();
    for (let i = 0; i < totalFrames; i++) {
      const x = frameToPixelX(i, totalFrames, canvasWidth);
      const y = mmToPixel(redMm[i] ?? (deepMm[i] ?? 24.0) + 12.0, canvasHeight, maxDepthMm);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // 8. Brush Indicator Circle
    if (cursorPos) {
      ctx.strokeStyle = toolMode === 'pen' ? 'rgba(255, 68, 68, 0.8)' : 'rgba(56, 189, 248, 0.8)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(cursorPos.x, cursorPos.y, brushRadius * zoom, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [canvasWidth, canvasHeight, superficialMm, deepMm, redMm, timestamps, cursorPos, brushRadius, zoom, toolMode]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Apply brush sculpting with quadratic smooth falloff
  const applyDraw = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;

      const totalFrames = timestamps.length;
      if (totalFrames === 0) return;

      const frameIdx = pixelXToFrame(x, totalFrames, canvasWidth);
      const targetDepthMm = pixelToMm(y, canvasHeight, maxDepthMm);

      const framesPerPx = totalFrames / canvasWidth;
      const brushFrames = Math.max(2, Math.round(brushRadius * framesPerPx * 1.5));

      const startIdx = Math.max(0, frameIdx - brushFrames);
      const endIdx = Math.min(totalFrames - 1, frameIdx + brushFrames);

      const updated = [...redMm];
      for (let i = startIdx; i <= endIdx; i++) {
        const dist = Math.abs(i - frameIdx) / (brushFrames + 1e-5);
        const weight = Math.max(0, 1 - dist * dist); // Smooth quadratic falloff

        if (toolMode === 'pen') {
          updated[i] = (1 - weight) * updated[i] + weight * targetDepthMm;
        } else {
          // Reset towards resting anatomical baseline
          const baseline = (deepMm[i] ?? 24.0) + 6.0;
          updated[i] = (1 - weight) * updated[i] + weight * baseline;
        }
      }

      onRedCurveChange(updated);
    },
    [timestamps, canvasWidth, canvasHeight, brushRadius, redMm, deepMm, toolMode, onRedCurveChange]
  );

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    isDrawingRef.current = true;
    applyDraw(e.clientX, e.clientY);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setCursorPos({ x, y });

    const totalFrames = timestamps.length;
    if (totalFrames > 0 && onHoverData) {
      const fIdx = pixelXToFrame(x, totalFrames, canvasWidth);
      const tSec = timestamps[fIdx] ?? 0;
      const dMm = pixelToMm(y, canvasHeight, maxDepthMm);
      const sup = superficialMm[fIdx] ?? 12;
      const deep = redMm[fIdx] ?? deepMm[fIdx] ?? 24;
      const thick = Math.max(0, deep - sup);
      onHoverData(tSec, fIdx + 1, dMm, thick);
    }

    if (isDrawingRef.current) {
      applyDraw(e.clientX, e.clientY);
    }
  };

  const handleMouseUp = () => {
    isDrawingRef.current = false;
  };

  const handleMouseLeave = () => {
    isDrawingRef.current = false;
    setCursorPos(null);
  };

  return (
    <div className="relative overflow-x-auto overflow-y-hidden border border-[#334155] rounded-lg bg-[#0b1120] shadow-2xl">
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className="cursor-crosshair block"
      />
    </div>
  );
};
