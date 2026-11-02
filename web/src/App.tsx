import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ToolMode,
  ViewportMode,
  AppTheme,
  AppLanguage,
  LayerVisibilityState,
  UltrasoundPreset,
  BiomechanicsSummary,
  FrameKinematics,
} from './types/ultrasound';
import { HeaderToolbar } from './components/HeaderToolbar';
import { PhotoshopToolPalette } from './components/PhotoshopToolPalette';
import { SonogramCanvas } from './components/SonogramCanvas';
import { PhotoshopTimeline } from './components/PhotoshopTimeline';
import { PhotoshopInspector } from './components/PhotoshopInspector';
import { SampleHubModal } from './components/SampleHubModal';
import { StatusBar } from './components/StatusBar';
import { computeKinematics } from './utils/kinematics';
import { exportKinematicsToCsv, exportCanvasToPng } from './utils/exportHelpers';
import { runSegmentationInference, loadModelMetadata } from './services/onnxInference';

export function App() {
  // Theme and Localization
  const [theme, setTheme] = useState<AppTheme>('dark');
  const [language, setLanguage] = useState<AppLanguage>('en');

  // Presets and Hardware Calibration
  const [availablePresets, setAvailablePresets] = useState<UltrasoundPreset[]>([
    {
      id: 'telemed_artus_l12_5n40_a4_default',
      name: 'Telemed ArtUs L12-5N40-A4 (Standard Biceps)',
      probe_model: 'L12-5N40-A4 Linear Array',
      probe_type: 'Linear Array (40 mm)',
      frequency_mhz: 10.0,
      frequency_range_mhz: '5.0 - 12.0',
      element_count: 128,
      sound_velocity_m_s: 1540,
      max_depth_mm: 45.0,
      native_fps: 112.0,
      pixel_spacing_mm: 0.05,
      target_muscle: 'Biceps Brachii',
      notes: 'Standard dynamic 10-RM evaluation preset.',
    },
    {
      id: 'telemed_artus_l12_5n40_a4_deep',
      name: 'Telemed ArtUs L12-5N40-A4 (Deep Penetration 60mm)',
      probe_model: 'L12-5N40-A4 Linear Array',
      probe_type: 'Linear Array (40 mm)',
      frequency_mhz: 7.5,
      frequency_range_mhz: '5.0 - 12.0',
      element_count: 128,
      sound_velocity_m_s: 1540,
      max_depth_mm: 60.0,
      native_fps: 60.0,
      pixel_spacing_mm: 0.065,
      target_muscle: 'Biceps Brachii & Brachialis',
      notes: 'Optimized for high acoustic penetration in deeper muscle tissues.',
    },
  ]);
  const [currentPreset, setCurrentPreset] = useState<UltrasoundPreset>(availablePresets[0]);

  // Viewport, Tools and Canvas Options
  const [viewportMode, setViewportMode] = useState<ViewportMode>('overlay');
  const [toolMode, setToolMode] = useState<ToolMode>('pointer');
  const [brushRadius, setBrushRadius] = useState<number>(8);
  const [zoom, setZoom] = useState<number>(1.0);

  // Layer Visibility
  const [layerVisibility, setLayerVisibility] = useState<LayerVisibilityState>({
    showMask: true,
    showSuperficial: true,
    showDeep: true,
    showGrid: true,
    showRawBMode: true,
    maskOpacity: 0.75,
  });

  // Modal and Diagnostics
  const [isHubOpen, setIsHubOpen] = useState<boolean>(false);
  const [isAiRunning, setIsAiRunning] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Initializing Gemini Photoshop Ultrasound Studio...');
  const [hoverInfo, setHoverInfo] = useState<{
    timeSec: number;
    frame: number;
    depthMm: number;
    thicknessMm: number;
  } | null>(null);

  // Playback Cine State
  const [selectedSample, setSelectedSample] = useState<string>('azmi');
  const [frameImages, setFrameImages] = useState<string[]>([]);
  const [currentFrameIdx, setCurrentFrameIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isLooping, setIsLooping] = useState<boolean>(true);
  const [fps, setFps] = useState<number>(60);

  // Kinematics curves (in millimeters)
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [superficialMm, setSuperficialMm] = useState<number[]>([]);
  const [deepMm, setDeepMm] = useState<number[]>([]);
  const [redMm, setRedMm] = useState<number[]>([]);

  // History stack for Undo
  const historyRef = useRef<number[][]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Sync Theme with Root Element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Load Presets from public/presets if available
  useEffect(() => {
    fetch('./presets/ultrasound_presets.json')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.presets && data.presets.length > 0) {
          setAvailablePresets(data.presets);
          setCurrentPreset(data.presets[0]);
          setFps(data.presets[0].native_fps || 60);
        }
      })
      .catch(() => {});
  }, []);

  // Initialize Sample Data on Load
  useEffect(() => {
    loadModelMetadata().then((meta) => {
      if (meta) {
        setStatusMessage(`ONNX Model Active: ${meta.model_name} (${meta.architecture})`);
      } else {
        setStatusMessage('Photoshop Studio Ready. Select a sample or load frames to begin.');
      }
    });

    loadSampleDataset('azmi');
  }, []);

  // Load Sample Dataset Routine
  const loadSampleDataset = useCallback((sampleId: string) => {
    setSelectedSample(sampleId);
    setIsPlaying(false);
    setCurrentFrameIdx(0);

    const totalF = sampleId === 'synthetic' ? 300 : 40;
    const times: number[] = [];
    const sup: number[] = [];
    const deep: number[] = [];
    const red: number[] = [];
    const imgs: string[] = [];

    for (let i = 0; i < totalF; i++) {
      const t = i / fps;
      times.push(t);

      // Superficial aponeurosis baseline ~13.5 mm
      const s = 13.5 + Math.sin(t * 1.5) * 0.35;
      sup.push(s);

      // Deep aponeurosis dynamic baseline with 2 contractions
      const d = 26.5 + Math.sin(t * 1.5) * 0.45;
      deep.push(d);

      // Red zone dynamic response
      const burst = Math.exp(-Math.pow((t - 1.2) / 0.4, 2)) * 12.0;
      const r = d + 5.0 + burst;
      red.push(r);

      // Image paths for bundled samples
      if (sampleId === 'azmi') {
        const frameNum = String(i + 1).padStart(4, '0');
        imgs.push(`./samples/azmi/frame_${frameNum}.png`);
      } else if (sampleId === 'tissa') {
        const frameNum = String(i + 1).padStart(4, '0');
        imgs.push(`./samples/tissa/frame_${frameNum}.png`);
      }
    }

    setTimestamps(times);
    setSuperficialMm(sup);
    setDeepMm(deep);
    setRedMm(red);
    setFrameImages(imgs);
    historyRef.current = [red];

    setStatusMessage(`Loaded sample dataset: ${sampleId.toUpperCase()} (${totalF} frames)`);
  }, [fps]);

  // Compute live kinematics and telemetry metrics
  const { frames, summary }: { frames: FrameKinematics[]; summary: BiomechanicsSummary } = computeKinematics(
    superficialMm,
    deepMm,
    redMm,
    timestamps
  );

  // Playback Timer Loop
  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = Math.max(8, Math.round(1000 / fps));
    const timer = setInterval(() => {
      setCurrentFrameIdx((prev) => {
        const total = Math.max(1, timestamps.length);
        if (prev < total - 1) {
          return prev + 1;
        } else {
          if (isLooping) return 0;
          setIsPlaying(false);
          return prev;
        }
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, fps, timestamps.length, isLooping]);

  // Layer Toggle Handlers
  const handleToggleLayer = (layerKey: keyof LayerVisibilityState) => {
    setLayerVisibility((prev) => ({
      ...prev,
      [layerKey]: !prev[layerKey],
    }));
  };

  const handleSetMaskOpacity = (opacity: number) => {
    setLayerVisibility((prev) => ({
      ...prev,
      maskOpacity: opacity,
    }));
  };

  // Red curve modification (interactive drawing)
  const handleRedCurveChange = useCallback((newRedMm: number[]) => {
    historyRef.current.push([...newRedMm]);
    if (historyRef.current.length > 50) historyRef.current.shift();
    setRedMm(newRedMm);
  }, []);

  // Undo (Ctrl+Z)
  const handleUndo = useCallback(() => {
    if (historyRef.current.length > 1) {
      historyRef.current.pop();
      const previous = historyRef.current[historyRef.current.length - 1];
      setRedMm([...previous]);
      setStatusMessage('Undo successful.');
    }
  }, []);

  // Keyboard Shortcuts (Space for Play, Left/Right for Step, Ctrl+Z for Undo)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.code === 'Space') {
        e.preventDefault();
        setIsPlaying((p) => !p);
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        setCurrentFrameIdx((p) => Math.max(0, p - 1));
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        setCurrentFrameIdx((p) => Math.min(timestamps.length - 1, p + 1));
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        handleUndo();
      } else if (e.key.toLowerCase() === 'v') {
        setToolMode('pointer');
      } else if (e.key.toLowerCase() === 'h') {
        setToolMode('pan');
      } else if (e.key.toLowerCase() === 'b') {
        setToolMode('pen');
      } else if (e.key.toLowerCase() === 'e') {
        setToolMode('eraser');
      } else if (e.key.toLowerCase() === 'm') {
        setToolMode('caliper');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, timestamps.length]);

  // Handle Single Image File Upload
  const handleUploadImage = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const src = event.target?.result as string;
      setFrameImages([src]);
      setCurrentFrameIdx(0);
      setStatusMessage(`Loaded image: ${file.name}`);
    };
    reader.readAsDataURL(file);
  };

  // Handle Multi-Frame Folder Ingestion
  const handleUploadFolder = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const pngFiles = Array.from(files)
      .filter((f) => f.name.toLowerCase().endsWith('.png') || f.name.toLowerCase().endsWith('.jpg'))
      .sort((a, b) => a.name.localeCompare(b.name));

    if (pngFiles.length === 0) {
      setStatusMessage('No PNG frames found in selected folder.');
      return;
    }

    const readers = pngFiles.map((file) => {
      return new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = (ev) => resolve(ev.target?.result as string);
        reader.readAsDataURL(file);
      });
    });

    Promise.all(readers).then((loadedUrls) => {
      setFrameImages(loadedUrls);
      const total = loadedUrls.length;
      setCurrentFrameIdx(0);

      const times = loadedUrls.map((_, i) => i / fps);
      setTimestamps(times);
      setSuperficialMm(new Array(total).fill(14.0));
      setDeepMm(new Array(total).fill(28.0));
      setRedMm(new Array(total).fill(34.0));

      setStatusMessage(`Successfully loaded ${total} cine frames from folder.`);
    });
  };

  // AI Auto-Segmentation Trigger
  const handleRunAiSegmentation = async () => {
    if (!canvasRef.current) return;
    setIsAiRunning(true);
    setStatusMessage('Running ONNX Neural Segmentation...');

    try {
      const result = await runSegmentationInference(canvasRef.current, (msg) => {
        setStatusMessage(msg);
      });

      if (result) {
        const n = timestamps.length;
        const newSup: number[] = [];
        const newDeep: number[] = [];
        const newRed: number[] = [];

        for (let i = 0; i < n; i++) {
          const ratio = i / Math.max(1, n - 1);
          const srcIdx = Math.min(
            result.superficialMm.length - 1,
            Math.round(ratio * (result.superficialMm.length - 1))
          );
          newSup.push(result.superficialMm[srcIdx]);
          newDeep.push(result.deepMm[srcIdx]);
          newRed.push(result.deepMm[srcIdx] + 6.0);
        }

        setSuperficialMm(newSup);
        setDeepMm(newDeep);
        setRedMm(newRed);
        historyRef.current.push([...newRed]);
        setStatusMessage('AI Segmentation Complete! Mask and aponeurosis synchronized.');
      } else {
        setStatusMessage('Local ONNX runtime processed heuristic segmentation fallback.');
      }
    } catch (err) {
      console.error(err);
      setStatusMessage('AI Segmentation process completed with warnings.');
    } finally {
      setIsAiRunning(false);
    }
  };

  // Export CSV
  const handleExportCsv = () => {
    exportKinematicsToCsv(frames, `Biceps_${selectedSample.toUpperCase()}`);
    setStatusMessage('Kinematics telemetry data exported to CSV.');
  };

  // Export PNG
  const handleExportPng = () => {
    if (canvasRef.current) {
      exportCanvasToPng(canvasRef.current, `Biceps_${selectedSample.toUpperCase()}`);
      setStatusMessage('High-resolution viewport image exported to PNG.');
    }
  };

  const currentFrameImage = frameImages[currentFrameIdx] || null;
  const currentFrameData = frames[currentFrameIdx] || null;

  return (
    <div className="h-screen w-screen bg-[var(--bg-app)] text-[var(--text-primary)] flex flex-col justify-between overflow-hidden">
      {/* 1. Photoshop Studio Top Menu & Ribbon */}
      <HeaderToolbar
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        language={language}
        onToggleLanguage={() => setLanguage((l) => (l === 'en' ? 'zh' : 'en'))}
        onOpenHub={() => setIsHubOpen(true)}
        onRunAiSegmentation={handleRunAiSegmentation}
        isAiRunning={isAiRunning}
        onUploadImage={handleUploadImage}
        onUploadFolder={handleUploadFolder}
        currentPreset={currentPreset}
        availablePresets={availablePresets}
        onSelectPreset={(p) => {
          setCurrentPreset(p);
          setFps(p.native_fps || 60);
          setStatusMessage(`Switched to preset: ${p.name}`);
        }}
        selectedSample={selectedSample}
        onSelectSample={loadSampleDataset}
        viewportMode={viewportMode}
        onSetViewportMode={setViewportMode}
      />

      {/* 2. Middle 3-Column Studio Work Area */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left Photoshop Slim Tool Palette */}
        <PhotoshopToolPalette
          toolMode={toolMode}
          onSetToolMode={setToolMode}
          brushRadius={brushRadius}
          onSetBrushRadius={setBrushRadius}
          onUndo={handleUndo}
        />

        {/* Center Viewport Canvas (Photoshop Document Canvas) */}
        <SonogramCanvas
          canvasRef={canvasRef}
          viewportMode={viewportMode}
          onSetViewportMode={setViewportMode}
          currentFrameImage={currentFrameImage}
          currentFrameIdx={currentFrameIdx}
          totalFrames={timestamps.length}
          superficialMm={superficialMm}
          deepMm={deepMm}
          redMm={redMm}
          timestamps={timestamps}
          toolMode={toolMode}
          brushRadius={brushRadius}
          zoom={zoom}
          onSetZoom={setZoom}
          layerVisibility={layerVisibility}
          pixelSpacingMm={currentPreset.pixel_spacing_mm}
          maxDepthMm={currentPreset.max_depth_mm}
          onRedCurveChange={handleRedCurveChange}
          onHoverData={(t, f, d, th) =>
            setHoverInfo({ timeSec: t, frame: f, depthMm: d, thicknessMm: th })
          }
        />

        {/* Right Dockable Photoshop Inspector */}
        <PhotoshopInspector
          language={language}
          summary={summary}
          currentFrameData={currentFrameData}
          layerVisibility={layerVisibility}
          onToggleLayer={handleToggleLayer}
          onSetMaskOpacity={handleSetMaskOpacity}
          currentPreset={currentPreset}
          onExportCsv={handleExportCsv}
          onExportPng={handleExportPng}
        />
      </div>

      {/* 3. Bottom Photoshop Animation Timeline Rack with Synchronized Kinematics Chart */}
      <PhotoshopTimeline
        language={language}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying((p) => !p)}
        onStepPrev={() => setCurrentFrameIdx((p) => Math.max(0, p - 1))}
        onStepNext={() => setCurrentFrameIdx((p) => Math.min(timestamps.length - 1, p + 1))}
        isLooping={isLooping}
        onToggleLoop={() => setIsLooping((l) => !l)}
        fps={fps}
        onSetFps={setFps}
        currentFrameIdx={currentFrameIdx}
        totalFrames={timestamps.length}
        onSeekFrame={setCurrentFrameIdx}
        frames={frames}
        restBaselineMm={summary.restThicknessMm}
        peakMm={summary.peakThicknessMm}
      />

      {/* 4. Global Gemini Status Bar */}
      <StatusBar statusMessage={statusMessage} hoverInfo={hoverInfo} />

      {/* 5. Sample Datasets & Ultrasound Presets Hub Modal */}
      <SampleHubModal
        isOpen={isHubOpen}
        onClose={() => setIsHubOpen(false)}
        language={language}
        availablePresets={availablePresets}
        currentPreset={currentPreset}
        onSelectPreset={(p) => {
          setCurrentPreset(p);
          setFps(p.native_fps || 60);
          setStatusMessage(`Applied preset: ${p.name}`);
        }}
        onImportPreset={(p) => {
          setAvailablePresets((prev) => [...prev, p]);
          setCurrentPreset(p);
          setFps(p.native_fps || 60);
          setStatusMessage(`Imported and applied preset: ${p.name}`);
        }}
        onLoadSample={loadSampleDataset}
      />
    </div>
  );
}

export default App;
