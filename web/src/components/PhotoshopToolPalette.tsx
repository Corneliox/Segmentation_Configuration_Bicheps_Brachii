import React from 'react';
import {
  MousePointer,
  Hand,
  PenTool,
  Eraser,
  Ruler,
  ZoomIn,
  RotateCcw,
} from 'lucide-react';
import { ToolMode } from '../types/ultrasound';

interface PhotoshopToolPaletteProps {
  toolMode: ToolMode;
  onSetToolMode: (mode: ToolMode) => void;
  brushRadius: number;
  onSetBrushRadius: (r: number) => void;
  onUndo: () => void;
}

export const PhotoshopToolPalette: React.FC<PhotoshopToolPaletteProps> = ({
  toolMode,
  onSetToolMode,
  brushRadius,
  onSetBrushRadius,
  onUndo,
}) => {
  const tools: { id: ToolMode; icon: React.ReactNode; label: string; shortcut: string }[] = [
    { id: 'pointer', icon: <MousePointer size={16} />, label: 'Selection / Cursor', shortcut: 'V' },
    { id: 'pan', icon: <Hand size={16} />, label: 'Hand / Pan Tool', shortcut: 'H' },
    { id: 'pen', icon: <PenTool size={16} />, label: 'Aponeurosis Pen Brush', shortcut: 'B' },
    { id: 'eraser', icon: <Eraser size={16} />, label: 'Eraser Tool', shortcut: 'E' },
    { id: 'caliper', icon: <Ruler size={16} />, label: 'Ultrasound Caliper Gauge', shortcut: 'M' },
    { id: 'zoom', icon: <ZoomIn size={16} />, label: 'Zoom Tool', shortcut: 'Z' },
  ];

  return (
    <aside className="w-12 bg-[var(--bg-surface)] border-r border-[var(--border-subtle)] flex flex-col items-center py-3 gap-1.5 select-none z-10">
      {tools.map((t) => {
        const isActive = toolMode === t.id;
        return (
          <button
            key={t.id}
            onClick={() => onSetToolMode(t.id)}
            title={`${t.label} (${t.shortcut})`}
            className={`w-8 h-8 rounded-lg flex items-center justify-center transition relative group ${
              isActive
                ? 'bg-[var(--gemini-blue)] text-black font-bold shadow-md'
                : 'text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)]'
            }`}
          >
            {t.icon}
            {/* Tooltip on hover */}
            <span className="absolute left-10 ml-1 px-2 py-1 rounded bg-[var(--bg-card)] text-[var(--text-primary)] text-[10px] whitespace-nowrap shadow-xl border border-[var(--border-subtle)] opacity-0 group-hover:opacity-100 pointer-events-none transition z-30">
              {t.label} ({t.shortcut})
            </span>
          </button>
        );
      })}

      <div className="w-6 h-[1px] bg-[var(--border-subtle)] my-2" />

      {/* Undo Button */}
      <button
        onClick={onUndo}
        title="Undo (Ctrl+Z)"
        className="w-8 h-8 rounded-lg flex items-center justify-center text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition relative group"
      >
        <RotateCcw size={15} />
        <span className="absolute left-10 ml-1 px-2 py-1 rounded bg-[var(--bg-card)] text-[var(--text-primary)] text-[10px] whitespace-nowrap shadow-xl border border-[var(--border-subtle)] opacity-0 group-hover:opacity-100 pointer-events-none transition z-30">
          Undo (Ctrl+Z)
        </span>
      </button>

      {/* Brush Size Indicator if Pen or Eraser is active */}
      {(toolMode === 'pen' || toolMode === 'eraser') && (
        <div className="flex flex-col items-center mt-2 gap-1" title={`Brush Radius: ${brushRadius}px`}>
          <div
            className="rounded-full bg-[var(--gemini-coral)] border border-white/50 transition-all"
            style={{ width: `${Math.min(22, Math.max(6, brushRadius * 1.5))}px`, height: `${Math.min(22, Math.max(6, brushRadius * 1.5))}px` }}
          />
          <span className="text-[9px] font-mono text-[var(--text-muted)]">{brushRadius}px</span>
        </div>
      )}

      {/* Photoshop Foreground / Background Color Swatches */}
      <div className="mt-auto flex flex-col items-center relative w-8 h-8">
        {/* Secondary color swatch (Gemini Violet) */}
        <div
          className="w-4 h-4 rounded-sm border border-[var(--border-strong)] absolute bottom-0 right-0 shadow-sm"
          style={{ backgroundColor: 'var(--gemini-purple)' }}
          title="Secondary Fascia Guide (#d0bcff)"
        />
        {/* Primary color swatch (Gemini Blue) */}
        <div
          className="w-4 h-4 rounded-sm border border-[var(--border-strong)] absolute top-0 left-0 shadow-sm"
          style={{ backgroundColor: 'var(--gemini-blue)' }}
          title="Primary Aponeurosis Guide (#a8c7fa)"
        />
      </div>
    </aside>
  );
};
