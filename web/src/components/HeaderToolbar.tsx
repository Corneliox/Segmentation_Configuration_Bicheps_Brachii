import React from 'react';
import {
  Sparkles,
  Sun,
  Moon,
  Globe,
  Database,
  Cpu,
  FolderOpen,
  Image as ImageIcon,
  CheckCircle2,
  Settings,
  ChevronDown,
  Layers,
  Activity,
  Sliders,
} from 'lucide-react';
import { AppTheme, AppLanguage, UltrasoundPreset, ViewportMode } from '../types/ultrasound';
import { TRANSLATIONS } from '../utils/translations';

interface HeaderToolbarProps {
  theme: AppTheme;
  onToggleTheme: () => void;
  language: AppLanguage;
  onToggleLanguage: () => void;
  onOpenHub: () => void;
  onRunAiSegmentation: () => void;
  isAiRunning: boolean;
  onUploadImage: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onUploadFolder: (e: React.ChangeEvent<HTMLInputElement>) => void;
  currentPreset: UltrasoundPreset;
  availablePresets: UltrasoundPreset[];
  onSelectPreset: (preset: UltrasoundPreset) => void;
  selectedSample: string;
  onSelectSample: (sampleId: string) => void;
  viewportMode: ViewportMode;
  onSetViewportMode: (mode: ViewportMode) => void;
}

export const HeaderToolbar: React.FC<HeaderToolbarProps> = ({
  theme,
  onToggleTheme,
  language,
  onToggleLanguage,
  onOpenHub,
  onRunAiSegmentation,
  isAiRunning,
  onUploadImage,
  onUploadFolder,
  currentPreset,
  availablePresets,
  onSelectPreset,
  selectedSample,
  onSelectSample,
  viewportMode,
  onSetViewportMode,
}) => {
  const t = TRANSLATIONS[language];

  return (
    <header className="border-b border-[var(--border-subtle)] bg-[var(--bg-surface)] text-[var(--text-primary)] select-none">
      {/* Top Photoshop Menu Line */}
      <div className="flex items-center justify-between px-3 py-1.5 text-xs border-b border-[var(--border-subtle)]/60 bg-[var(--bg-surface)]/80">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 font-bold tracking-tight">
            <span className="p-1 rounded-md bg-[var(--bg-card)] text-[var(--gemini-blue)]">
              <Sparkles size={14} className="text-[var(--gemini-blue)]" />
            </span>
            <span className="font-extrabold text-xs tracking-wider">
              GEMINI <span className="gemini-gradient-text">ULTRASOUND</span>
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[var(--text-muted)] font-mono">
              {t.studio_badge}
            </span>
          </div>

          {/* Photoshop Top Menus */}
          <nav className="hidden md:flex items-center gap-1 text-[var(--text-secondary)]">
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_file}
            </button>
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_edit}
            </button>
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_presets}
            </button>
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_layers}
            </button>
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_view}
            </button>
            <button className="px-2 py-0.5 rounded hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition">
              {t.menu_export}
            </button>
          </nav>
        </div>

        {/* Right Top Status & Theme/Lang */}
        <div className="flex items-center gap-2">
          {/* Active Preset Tag */}
          <div className="hidden lg:flex items-center gap-1 text-[11px] text-[var(--text-muted)] bg-[var(--bg-card)] px-2 py-0.5 rounded border border-[var(--border-subtle)]">
            <Activity size={12} className="text-[var(--gemini-emerald)]" />
            <span>{currentPreset.probe_model}</span>
            <span className="text-[var(--gemini-blue)] font-mono">({currentPreset.pixel_spacing_mm} mm/px)</span>
          </div>

          {/* Language Switcher */}
          <button
            onClick={onToggleLanguage}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-[11px] font-medium transition"
            title="Toggle English / 繁體中文"
          >
            <Globe size={12} className="text-[var(--gemini-blue)]" />
            <span>{language === 'en' ? 'EN' : '繁中'}</span>
          </button>

          {/* Theme Switcher */}
          <button
            onClick={onToggleTheme}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-[11px] font-medium transition"
            title="Toggle Gemini Dark / Light"
          >
            {theme === 'dark' ? (
              <>
                <Moon size={12} className="text-[var(--gemini-purple)]" />
                <span>Night</span>
              </>
            ) : (
              <>
                <Sun size={12} className="text-[var(--gemini-coral)]" />
                <span>Day</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Studio Action Ribbon */}
      <div className="px-3 py-2 flex flex-wrap items-center justify-between gap-3">
        {/* Left: Preset Selector, Sample Trials & Ingestion */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Preset Selector */}
          <div className="flex items-center gap-1.5 bg-[var(--bg-card)] px-2.5 py-1.5 rounded-lg border border-[var(--border-subtle)] text-xs">
            <Settings size={14} className="text-[var(--gemini-blue)]" />
            <span className="text-[var(--text-muted)] font-medium">Preset:</span>
            <select
              value={currentPreset.id}
              onChange={(e) => {
                const found = availablePresets.find((p) => p.id === e.target.value);
                if (found) onSelectPreset(found);
              }}
              className="bg-transparent text-[var(--text-primary)] font-semibold outline-none cursor-pointer text-xs pr-1"
            >
              {availablePresets.map((p) => (
                <option key={p.id} value={p.id} className="bg-[var(--bg-card)] text-[var(--text-primary)]">
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Sample Trial Selector */}
          <div className="flex items-center gap-1.5 bg-[var(--bg-card)] px-2.5 py-1.5 rounded-lg border border-[var(--border-subtle)] text-xs">
            <Database size={14} className="text-[var(--gemini-purple)]" />
            <span className="text-[var(--text-muted)] font-medium">Sample:</span>
            <select
              value={selectedSample}
              onChange={(e) => onSelectSample(e.target.value)}
              className="bg-transparent text-[var(--text-primary)] font-semibold outline-none cursor-pointer text-xs pr-1"
            >
              <option value="azmi" className="bg-[var(--bg-card)] text-[var(--text-primary)]">
                {t.sample_azmi}
              </option>
              <option value="tissa" className="bg-[var(--bg-card)] text-[var(--text-primary)]">
                {t.sample_tissa}
              </option>
              <option value="synthetic" className="bg-[var(--bg-card)] text-[var(--text-primary)]">
                {t.sample_synthetic}
              </option>
            </select>
          </div>

          {/* Viewport Mode Switcher (Overlay / Side-by-Side / Raw / Filmstrip) */}
          <div className="flex items-center bg-[var(--bg-card)] p-0.5 rounded-lg border border-[var(--border-subtle)] text-xs">
            <button
              onClick={() => onSetViewportMode('overlay')}
              className={`px-2 py-1 rounded-md text-xs font-semibold transition ${
                viewportMode === 'overlay'
                  ? 'bg-[var(--gemini-blue)] text-black shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              Overlay
            </button>
            <button
              onClick={() => onSetViewportMode('side_by_side')}
              className={`px-2 py-1 rounded-md text-xs font-semibold transition ${
                viewportMode === 'side_by_side'
                  ? 'bg-[var(--gemini-blue)] text-black shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              Dual View
            </button>
            <button
              onClick={() => onSetViewportMode('raw')}
              className={`px-2 py-1 rounded-md text-xs font-semibold transition ${
                viewportMode === 'raw'
                  ? 'bg-[var(--gemini-blue)] text-black shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              Raw
            </button>
            <button
              onClick={() => onSetViewportMode('filmstrip')}
              className={`px-2 py-1 rounded-md text-xs font-semibold transition ${
                viewportMode === 'filmstrip'
                  ? 'bg-[var(--gemini-blue)] text-black shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              Filmstrip
            </button>
          </div>

          {/* Open Image File Button */}
          <label className="cursor-pointer flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] text-xs font-semibold border border-[var(--border-subtle)] transition">
            <ImageIcon size={14} className="text-[var(--gemini-emerald)]" />
            <span>{t.btn_upload_image}</span>
            <input type="file" accept="image/*" className="hidden" onChange={onUploadImage} />
          </label>

          {/* Open Folder / Cine Frames */}
          <label className="cursor-pointer flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] text-xs font-semibold border border-[var(--border-subtle)] transition">
            <FolderOpen size={14} className="text-[var(--gemini-yellow)]" />
            <span>{t.btn_upload_folder}</span>
            <input
              type="file"
              // @ts-ignore: webkitdirectory is standard for folder picker
              webkitdirectory=""
              directory=""
              multiple
              className="hidden"
              onChange={onUploadFolder}
            />
          </label>
        </div>

        {/* Right: Hub Modal & AI Run Action */}
        <div className="flex items-center gap-2">
          {/* Sample Datasets & Presets Hub Modal Trigger */}
          <button
            onClick={onOpenHub}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-xs font-bold text-[var(--gemini-purple)] transition"
          >
            <Database size={14} />
            <span>{t.btn_sample_hub}</span>
          </button>

          {/* AI Auto-Segmentation Button */}
          <button
            onClick={onRunAiSegmentation}
            disabled={isAiRunning}
            className="flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-extrabold text-black transition shadow-lg disabled:opacity-50"
            style={{
              background: 'linear-gradient(135deg, #a8c7fa 0%, #d0bcff 100%)',
            }}
          >
            <Sparkles size={14} className={isAiRunning ? 'animate-spin' : ''} />
            <span>{isAiRunning ? t.btn_running_ai : t.btn_run_ai}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
