import React, { useState } from 'react';
import {
  X,
  Database,
  Sliders,
  Download,
  Upload,
  CheckCircle2,
  FileCode,
  HardDrive,
  Info,
} from 'lucide-react';
import { UltrasoundPreset, AppLanguage } from '../types/ultrasound';
import { TRANSLATIONS } from '../utils/translations';

interface SampleHubModalProps {
  isOpen: boolean;
  onClose: () => void;
  language: AppLanguage;
  availablePresets: UltrasoundPreset[];
  currentPreset: UltrasoundPreset;
  onSelectPreset: (preset: UltrasoundPreset) => void;
  onImportPreset: (preset: UltrasoundPreset) => void;
  onLoadSample: (sampleId: string) => void;
}

export const SampleHubModal: React.FC<SampleHubModalProps> = ({
  isOpen,
  onClose,
  language,
  availablePresets,
  currentPreset,
  onSelectPreset,
  onImportPreset,
  onLoadSample,
}) => {
  const [activeTab, setActiveTab] = useState<'datasets' | 'presets'>('datasets');
  const [remoteUrl, setRemoteUrl] = useState(
    'https://github.com/Antigravity/Probefix/releases/download/v1.0.0/biceps_sample_trials.zip'
  );
  const [importMessage, setImportMessage] = useState<string | null>(null);

  if (!isOpen) return null;
  const t = TRANSLATIONS[language];

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = JSON.parse(text);
        const preset: UltrasoundPreset = parsed.presets ? parsed.presets[0] : parsed;
        onImportPreset(preset);
        setImportMessage(`Successfully imported preset: "${preset.name}"`);
      } catch (err) {
        setImportMessage('Error: Failed to parse preset JSON file.');
      }
    };
    reader.readAsText(file);
  };

  const exportPresetTemplate = () => {
    const template: UltrasoundPreset = {
      id: 'custom_artus_spec',
      name: 'Custom Transducer Specification',
      probe_model: 'Telemed ArtUs L12-5N40-A4',
      probe_type: 'Linear Array 40 mm',
      frequency_mhz: 10.0,
      frequency_range_mhz: '5.0 - 12.0',
      element_count: 128,
      sound_velocity_m_s: 1540,
      max_depth_mm: 45.0,
      native_fps: 112.0,
      pixel_spacing_mm: 0.05,
      target_muscle: 'Biceps Brachii',
      notes: 'User custom calibration specifications.',
    };
    const blob = new Blob([JSON.stringify(template, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ultrasound_preset_template.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[var(--bg-surface)] border border-[var(--border-strong)] rounded-2xl max-w-2xl w-full shadow-2xl overflow-hidden flex flex-col max-h-[85vh] text-[var(--text-primary)]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border-subtle)] bg-[var(--bg-card)]">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-[var(--gemini-blue)]/20 text-[var(--gemini-blue)]">
              <Database size={18} />
            </span>
            <h2 className="font-extrabold text-sm">{t.hub_title}</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center hover:bg-[var(--bg-card-hover)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="flex border-b border-[var(--border-subtle)] bg-[var(--bg-card)]/50 px-5 text-xs font-bold">
          <button
            onClick={() => setActiveTab('datasets')}
            className={`py-3 px-4 border-b-2 transition ${
              activeTab === 'datasets'
                ? 'border-[var(--gemini-blue)] text-[var(--gemini-blue)]'
                : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
          >
            {t.hub_tab_datasets}
          </button>
          <button
            onClick={() => setActiveTab('presets')}
            className={`py-3 px-4 border-b-2 transition ${
              activeTab === 'presets'
                ? 'border-[var(--gemini-purple)] text-[var(--gemini-purple)]'
                : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
          >
            {t.hub_tab_presets}
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs flex-1">
          {activeTab === 'datasets' ? (
            <div className="space-y-3">
              {/* Card 1: Subject Azmi */}
              <div className="p-3.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] flex items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 font-bold text-sm">
                    <span>Subject Azmi (10-RM Flexion)</span>
                    <span className="px-2 py-0.5 rounded-full bg-[var(--gemini-emerald)]/20 text-[var(--gemini-emerald)] text-[10px] font-mono">
                      858 frames (Cine)
                    </span>
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-1">
                    Continuous arm dynamic contraction cine loop | High dynamic range RF sampling | Clear aponeurosis.
                  </p>
                </div>
                <button
                  onClick={() => {
                    onLoadSample('azmi');
                    onClose();
                  }}
                  className="px-3.5 py-1.5 rounded-lg bg-[var(--gemini-blue)] text-black font-extrabold hover:opacity-90 transition whitespace-nowrap shadow-sm"
                >
                  {t.hub_btn_load}
                </button>
              </div>

              {/* Card 2: Subject Tissa */}
              <div className="p-3.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] flex items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 font-bold text-sm">
                    <span>Subject Tissa (10-RM Flexion)</span>
                    <span className="px-2 py-0.5 rounded-full bg-[var(--gemini-emerald)]/20 text-[var(--gemini-emerald)] text-[10px] font-mono">
                      918 frames (Cine)
                    </span>
                  </div>
                  <p className="text-[11px] text-[var(--text-secondary)] mt-1">
                    Complete concentric and eccentric muscle contraction phases | Pronounced fascia motion.
                  </p>
                </div>
                <button
                  onClick={() => {
                    onLoadSample('tissa');
                    onClose();
                  }}
                  className="px-3.5 py-1.5 rounded-lg bg-[var(--gemini-blue)] text-black font-extrabold hover:opacity-90 transition whitespace-nowrap shadow-sm"
                >
                  {t.hub_btn_load}
                </button>
              </div>

              {/* Cloud Asset Packaging Box */}
              <div className="p-3.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] space-y-2 mt-4">
                <div className="font-bold flex items-center gap-1.5">
                  <Download size={14} className="text-[var(--gemini-purple)]" />
                  <span>Cloud Dataset Download Link (.zip)</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={remoteUrl}
                    onChange={(e) => setRemoteUrl(e.target.value)}
                    className="flex-1 px-3 py-1.5 rounded-lg bg-[var(--bg-surface)] border border-[var(--border-subtle)] text-xs font-mono outline-none"
                  />
                  <a
                    href={remoteUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="px-3 py-1.5 rounded-lg bg-[var(--bg-card-hover)] hover:bg-[var(--border-subtle)] border border-[var(--border-subtle)] font-bold transition whitespace-nowrap flex items-center gap-1"
                  >
                    <Download size={13} />
                    <span>Download</span>
                  </a>
                </div>
              </div>

              {/* Hosting Advice Notice */}
              <div className="p-3 rounded-xl bg-[var(--bg-card)]/50 border border-[var(--border-subtle)] text-[11px] text-[var(--text-muted)] flex items-start gap-2">
                <Info size={14} className="text-[var(--gemini-blue)] shrink-0 mt-0.5" />
                <p>{t.hub_hosting_advice}</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Presets List */}
              <div className="space-y-2">
                {availablePresets.map((p) => {
                  const isCurrent = p.id === currentPreset.id;
                  return (
                    <div
                      key={p.id}
                      className={`p-3.5 rounded-xl border transition ${
                        isCurrent
                          ? 'bg-[var(--gemini-purple)]/10 border-[var(--gemini-purple)]'
                          : 'bg-[var(--bg-card)] border-[var(--border-subtle)]'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="font-bold text-sm text-[var(--text-primary)]">{p.name}</span>
                          <span className="ml-2 text-[10px] text-[var(--text-muted)] font-mono">({p.probe_type})</span>
                        </div>
                        {isCurrent ? (
                          <span className="text-[10px] font-bold text-[var(--gemini-purple)] flex items-center gap-1">
                            <CheckCircle2 size={13} /> Active
                          </span>
                        ) : (
                          <button
                            onClick={() => onSelectPreset(p)}
                            className="px-2.5 py-1 rounded-lg bg-[var(--bg-surface)] hover:bg-[var(--bg-card-hover)] text-xs font-bold border border-[var(--border-subtle)] transition"
                          >
                            Apply
                          </button>
                        )}
                      </div>

                      <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-[var(--border-subtle)]/60 text-[11px] font-mono text-[var(--text-secondary)]">
                        <div>Freq: <strong className="text-[var(--gemini-blue)]">{p.frequency_mhz} MHz</strong></div>
                        <div>FPS: <strong className="text-[var(--gemini-emerald)]">{p.native_fps} FPS</strong></div>
                        <div>Scale: <strong className="text-[var(--gemini-yellow)]">{p.pixel_spacing_mm} mm/px</strong></div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Import Custom Preset File Section */}
              <div className="p-3.5 rounded-xl bg-[var(--bg-card)] border border-[var(--border-subtle)] space-y-2">
                <div className="font-bold flex items-center gap-1.5">
                  <Upload size={14} className="text-[var(--gemini-emerald)]" />
                  <span>{t.hub_import_preset}</span>
                </div>
                <p className="text-[11px] text-[var(--text-secondary)]">
                  Upload your ultrasound scanner preset specifications (.json or .yaml) to automatically calibrate depth, speed of sound, and sampling frequency.
                </p>
                <div className="flex items-center gap-2 pt-1">
                  <label className="cursor-pointer px-3.5 py-1.5 rounded-lg bg-[var(--gemini-emerald)] text-black font-bold text-xs hover:opacity-90 transition flex items-center gap-1.5 shadow-sm">
                    <Upload size={13} />
                    <span>Select Preset File (.json)</span>
                    <input type="file" accept=".json,.yaml,.yml" className="hidden" onChange={handleFileUpload} />
                  </label>

                  <button
                    onClick={exportPresetTemplate}
                    className="px-3.5 py-1.5 rounded-lg bg-[var(--bg-surface)] hover:bg-[var(--bg-card-hover)] border border-[var(--border-subtle)] text-xs font-semibold transition flex items-center gap-1.5"
                  >
                    <FileCode size={13} />
                    <span>{t.hub_export_template}</span>
                  </button>
                </div>
                {importMessage && (
                  <p className="text-[11px] font-mono text-[var(--gemini-blue)] pt-1">{importMessage}</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-[var(--border-subtle)] bg-[var(--bg-card)] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-[var(--bg-surface)] hover:bg-[var(--bg-card-hover)] text-xs font-bold border border-[var(--border-subtle)] transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
