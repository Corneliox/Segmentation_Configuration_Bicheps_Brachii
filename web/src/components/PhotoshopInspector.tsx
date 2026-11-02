import React from 'react';
import {
  Layers,
  Activity,
  Sliders,
  Eye,
  EyeOff,
  FileSpreadsheet,
  Download,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import {
  BiomechanicsSummary,
  FrameKinematics,
  LayerVisibilityState,
  UltrasoundPreset,
  AppLanguage,
} from '../types/ultrasound';
import { TRANSLATIONS } from '../utils/translations';

interface PhotoshopInspectorProps {
  language: AppLanguage;
  summary: BiomechanicsSummary;
  currentFrameData: FrameKinematics | null;
  layerVisibility: LayerVisibilityState;
  onToggleLayer: (layerKey: keyof LayerVisibilityState) => void;
  onSetMaskOpacity: (opacity: number) => void;
  currentPreset: UltrasoundPreset;
  onExportCsv: () => void;
  onExportPng: () => void;
}

export const PhotoshopInspector: React.FC<PhotoshopInspectorProps> = ({
  language,
  summary,
  currentFrameData,
  layerVisibility,
  onToggleLayer,
  onSetMaskOpacity,
  currentPreset,
  onExportCsv,
  onExportPng,
}) => {
  const t = TRANSLATIONS[language];

  const currentThickness = currentFrameData ? currentFrameData.muscleThicknessMm : summary.restThicknessMm;
  const isCurrentlyActive = currentFrameData ? currentFrameData.state === 'ACTIVE' : false;

  return (
    <aside className="w-80 bg-[var(--bg-surface)] border-l border-[var(--border-subtle)] flex flex-col h-full overflow-y-auto select-none text-[var(--text-primary)]">
      {/* 1. Layers Panel (Photoshop Layers) */}
      <div className="border-b border-[var(--border-subtle)] p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider">
            <Layers size={14} className="text-[var(--gemini-blue)]" />
            <span>{t.layers_title}</span>
          </div>
          <span className="text-[10px] text-[var(--text-muted)] font-mono">5 Layers</span>
        </div>

        {/* Opacity Slider for Mask */}
        <div className="flex items-center justify-between text-xs mb-3 bg-[var(--bg-card)] px-2.5 py-1.5 rounded-lg border border-[var(--border-subtle)]">
          <span className="text-[11px] text-[var(--text-secondary)]">{t.opacity_label}:</span>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min="0"
              max="100"
              value={Math.round(layerVisibility.maskOpacity * 100)}
              onChange={(e) => onSetMaskOpacity(Number(e.target.value) / 100)}
              className="w-20 h-1.5 accent-[var(--gemini-blue)] cursor-pointer"
            />
            <span className="text-[11px] font-mono text-[var(--gemini-blue)] w-8 text-right">
              {Math.round(layerVisibility.maskOpacity * 100)}%
            </span>
          </div>
        </div>

        {/* Layer Stack Items */}
        <div className="space-y-1 text-xs">
          {/* Layer: Biceps Mask */}
          <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] transition">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onToggleLayer('showMask')}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                {layerVisibility.showMask ? <Eye size={14} className="text-[var(--gemini-emerald)]" /> : <EyeOff size={14} />}
              </button>
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--gemini-emerald)]" />
              <span className="font-medium">{t.layer_mask}</span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">U-Net AI</span>
          </div>

          {/* Layer: Superficial Fascia */}
          <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] transition">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onToggleLayer('showSuperficial')}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                {layerVisibility.showSuperficial ? <Eye size={14} className="text-[var(--gemini-blue)]" /> : <EyeOff size={14} />}
              </button>
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--gemini-blue)]" />
              <span className="font-medium">{t.layer_superficial}</span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">Upper</span>
          </div>

          {/* Layer: Deep Fascia */}
          <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] transition">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onToggleLayer('showDeep')}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                {layerVisibility.showDeep ? <Eye size={14} className="text-[var(--gemini-purple)]" /> : <EyeOff size={14} />}
              </button>
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--gemini-purple)]" />
              <span className="font-medium">{t.layer_deep}</span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">Lower</span>
          </div>

          {/* Layer: Depth Grid */}
          <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] transition">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onToggleLayer('showGrid')}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                {layerVisibility.showGrid ? <Eye size={14} className="text-[var(--gemini-yellow)]" /> : <EyeOff size={14} />}
              </button>
              <div className="w-2.5 h-2.5 rounded-full bg-[var(--gemini-yellow)]" />
              <span className="font-medium">{t.layer_grid}</span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">10mm</span>
          </div>

          {/* Layer: Raw B-Mode */}
          <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] transition">
            <div className="flex items-center gap-2">
              <button
                onClick={() => onToggleLayer('showRawBMode')}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              >
                {layerVisibility.showRawBMode ? <Eye size={14} className="text-[var(--text-primary)]" /> : <EyeOff size={14} />}
              </button>
              <div className="w-2.5 h-2.5 rounded-full bg-slate-400" />
              <span className="font-medium">{t.layer_raw}</span>
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono">Base</span>
          </div>
        </div>
      </div>

      {/* 2. Biomechanics Telemetry Inspector */}
      <div className="border-b border-[var(--border-subtle)] p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider">
            <Activity size={14} className="text-[var(--gemini-emerald)]" />
            <span>{t.telemetry_title}</span>
          </div>
        </div>

        {/* State Banner with Aura */}
        <div
          className={`p-2.5 rounded-xl border text-center font-bold text-xs mb-3 transition shadow-md ${
            isCurrentlyActive
              ? 'bg-[var(--gemini-coral)]/15 border-[var(--gemini-coral)] text-[var(--gemini-coral)] shadow-[0_0_15px_rgba(242,139,130,0.2)]'
              : 'bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-secondary)]'
          }`}
        >
          {isCurrentlyActive ? t.status_active : t.status_relax}
        </div>

        {/* Telemetry Metric Cards */}
        <div className="space-y-2">
          {/* Card: Current Thickness */}
          <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)]">
            <div className="text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
              {t.metric_current}
            </div>
            <div className="text-2xl font-black font-mono text-[var(--gemini-blue)] mt-0.5">
              {currentThickness.toFixed(2)} <span className="text-xs font-normal text-[var(--text-muted)]">mm</span>
            </div>
          </div>

          {/* 2-Column: Rest vs Peak */}
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)]">
              <div className="text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
                {t.metric_rest}
              </div>
              <div className="text-lg font-extrabold font-mono text-[var(--gemini-yellow)] mt-0.5">
                {summary.restThicknessMm.toFixed(2)} <span className="text-[10px] font-normal text-[var(--text-muted)]">mm</span>
              </div>
            </div>

            <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)]">
              <div className="text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
                {t.metric_peak}
              </div>
              <div className="text-lg font-extrabold font-mono text-[var(--gemini-coral)] mt-0.5">
                {summary.peakThicknessMm.toFixed(2)} <span className="text-[10px] font-normal text-[var(--text-muted)]">mm</span>
              </div>
            </div>
          </div>

          {/* Delta & Strain Card */}
          <div className="p-2.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)]">
            <div className="text-[10px] text-[var(--text-muted)] font-bold uppercase tracking-wider">
              {t.metric_delta}
            </div>
            <div className="text-base font-extrabold font-mono text-[var(--gemini-emerald)] mt-0.5">
              +{summary.deltaThicknessMm.toFixed(2)} mm{' '}
              <span className="text-xs font-semibold">({summary.percentageIncrease.toFixed(1)}%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Ultrasound Transducer Hardware Specifications */}
      <div className="border-b border-[var(--border-subtle)] p-3">
        <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--text-primary)] uppercase tracking-wider mb-2">
          <Sliders size={14} className="text-[var(--gemini-purple)]" />
          <span>{t.specs_title}</span>
        </div>

        <div className="bg-[var(--bg-card)] p-2.5 rounded-xl border border-[var(--border-subtle)] space-y-1.5 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-[var(--text-muted)] font-sans">{t.spec_transducer}:</span>
            <span className="font-bold text-[var(--text-primary)]">{currentPreset.probe_model}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[var(--text-muted)] font-sans">{t.spec_freq}:</span>
            <span className="text-[var(--gemini-blue)]">{currentPreset.frequency_mhz} MHz</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[var(--text-muted)] font-sans">{t.spec_fps}:</span>
            <span className="text-[var(--gemini-emerald)]">{currentPreset.native_fps} FPS</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[var(--text-muted)] font-sans">{t.spec_sound_speed}:</span>
            <span className="text-[var(--gemini-yellow)]">{currentPreset.sound_velocity_m_s} m/s</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[var(--text-muted)] font-sans">{t.spec_calibration}:</span>
            <span className="text-[var(--gemini-purple)] font-bold">{currentPreset.pixel_spacing_mm} mm/px</span>
          </div>
        </div>
      </div>

      {/* 4. Action Export Buttons */}
      <div className="p-3 mt-auto space-y-2">
        <button
          onClick={onExportCsv}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-xs font-bold text-[var(--gemini-blue)] transition shadow-sm"
        >
          <FileSpreadsheet size={15} />
          <span>{t.btn_export_csv}</span>
        </button>

        <button
          onClick={onExportPng}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-xs font-bold text-[var(--text-primary)] transition shadow-sm"
        >
          <Download size={15} />
          <span>{t.btn_export_png}</span>
        </button>
      </div>
    </aside>
  );
};
