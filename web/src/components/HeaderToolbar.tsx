import React from 'react';
import { ToolMode } from '../types/ultrasound';
import {
  PenTool,
  Eraser,
  RotateCcw,
  Download,
  FileSpreadsheet,
  Cpu,
  Upload,
  ZoomIn,
} from 'lucide-react';

interface HeaderToolbarProps {
  toolMode: ToolMode;
  onSetToolMode: (mode: ToolMode) => void;
  brushRadius: number;
  onSetBrushRadius: (val: number) => void;
  zoom: number;
  onSetZoom: (val: number) => void;
  onUndo: () => void;
  onExportCsv: () => void;
  onExportPng: () => void;
  onRunAiSegmentation: () => void;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  isAiRunning: boolean;
}

export const HeaderToolbar: React.FC<HeaderToolbarProps> = ({
  toolMode,
  onSetToolMode,
  brushRadius,
  onSetBrushRadius,
  zoom,
  onSetZoom,
  onUndo,
  onExportCsv,
  onExportPng,
  onRunAiSegmentation,
  onFileUpload,
  isAiRunning,
}) => {
  return (
    <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-3 shadow-xl flex flex-wrap items-center justify-between gap-3">
      {/* Left: App Title and Upload */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 pr-3 border-r border-[#334155]">
          <span className="p-1.5 rounded-lg bg-[#0284c7] text-white">
            <Cpu size={18} />
          </span>
          <div>
            <h1 className="text-sm font-black text-white leading-tight">
              Biceps Brachii <span className="text-[#38bdf8]">US Studio</span>
            </h1>
            <p className="text-[10px] text-[#94a3b8]">AI Segmentation & Kinematics</p>
          </div>
        </div>

        {/* Upload Button */}
        <label className="cursor-pointer flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1e293b] hover:bg-[#334155] text-xs font-bold text-white border border-[#475569] transition">
          <Upload size={14} className="text-[#38bdf8]" />
          <span>Upload Image</span>
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onFileUpload}
          />
        </label>
      </div>

      {/* Middle: Drawing Tools and Sliders */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Pen Button */}
        <button
          onClick={() => onSetToolMode('pen')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition border ${
            toolMode === 'pen'
              ? 'bg-[#dc2626] border-[#ef4444] text-white shadow-lg'
              : 'bg-[#1e293b] border-[#334155] text-[#94a3b8] hover:bg-[#334155]'
          }`}
        >
          <PenTool size={14} />
          <span>Kuas Merah</span>
        </button>

        {/* Eraser Button */}
        <button
          onClick={() => onSetToolMode('eraser')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition border ${
            toolMode === 'eraser'
              ? 'bg-[#0284c7] border-[#38bdf8] text-white shadow-lg'
              : 'bg-[#1e293b] border-[#334155] text-[#94a3b8] hover:bg-[#334155]'
          }`}
        >
          <Eraser size={14} />
          <span>Penghapus</span>
        </button>

        {/* Brush Size Slider */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#1e293b] border border-[#334155]">
          <span className="text-[11px] text-[#94a3b8] font-bold">Kuas:</span>
          <input
            type="range"
            min={3}
            max={30}
            value={brushRadius}
            onChange={(e) => onSetBrushRadius(Number(e.target.value))}
            className="w-16 accent-[#00f2fe] cursor-pointer"
          />
          <span className="text-[11px] text-white font-mono w-4">{brushRadius}</span>
        </div>

        {/* Zoom Slider */}
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#1e293b] border border-[#334155]">
          <ZoomIn size={14} className="text-[#94a3b8]" />
          <span className="text-[11px] text-[#94a3b8] font-bold">Zoom:</span>
          <input
            type="range"
            min={0.8}
            max={3.0}
            step={0.1}
            value={zoom}
            onChange={(e) => onSetZoom(Number(e.target.value))}
            className="w-16 accent-[#38bdf8] cursor-pointer"
          />
          <span className="text-[11px] text-white font-mono w-7">{zoom.toFixed(1)}x</span>
        </div>

        {/* Undo Button */}
        <button
          onClick={onUndo}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[#1e293b] hover:bg-[#334155] text-xs font-bold text-white border border-[#334155] transition"
          title="Undo (Ctrl+Z)"
        >
          <RotateCcw size={13} />
          <span>Undo</span>
        </button>
      </div>

      {/* Right: AI Segmentation & Export Actions */}
      <div className="flex items-center gap-2">
        {/* Run AI Button */}
        <button
          onClick={onRunAiSegmentation}
          disabled={isAiRunning}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#059669] hover:bg-[#047857] text-xs font-bold text-white border border-[#10b981] shadow-lg transition disabled:opacity-50"
        >
          <Cpu size={14} className={isAiRunning ? 'animate-spin' : ''} />
          <span>{isAiRunning ? 'AI Segmenting...' : 'Auto Segment (AI)'}</span>
        </button>

        {/* Export CSV */}
        <button
          onClick={onExportCsv}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#1e293b] hover:bg-[#334155] text-xs font-bold text-white border border-[#334155] transition"
        >
          <FileSpreadsheet size={14} className="text-[#4ade80]" />
          <span>CSV</span>
        </button>

        {/* Export PNG */}
        <button
          onClick={onExportPng}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0284c7] hover:bg-[#0369a1] text-xs font-bold text-white border border-[#38bdf8] transition"
        >
          <Download size={14} />
          <span>Export PNG</span>
        </button>
      </div>
    </div>
  );
};
