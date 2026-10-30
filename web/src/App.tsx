import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ToolMode, BiomechanicsSummary, FrameKinematics } from './types/ultrasound';
import { HeaderToolbar } from './components/HeaderToolbar';
import { SonogramCanvas } from './components/SonogramCanvas';
import { KinematicsChart } from './components/KinematicsChart';
import { MetricsCards } from './components/MetricsCards';
import { StatusBar } from './components/StatusBar';
import { computeKinematics } from './utils/kinematics';
import { exportKinematicsToCsv, exportCanvasToPng } from './utils/exportHelpers';
import { runSegmentationInference, loadModelMetadata } from './services/onnxInference';

export function App() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [toolMode, setToolMode] = useState<ToolMode>('pen');
  const [brushRadius, setBrushRadius] = useState<number>(8);
  const [zoom, setZoom] = useState<number>(1.0);
  const [isAiRunning, setIsAiRunning] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Memuat Biceps Brachii Ultrasound Studio...');

  // Curves state (in millimeters)
  const totalDefaultFrames = 300;
  const [timestamps, setTimestamps] = useState<number[]>([]);
  const [superficialMm, setSuperficialMm] = useState<number[]>([]);
  const [deepMm, setDeepMm] = useState<number[]>([]);
  const [redMm, setRedMm] = useState<number[]>([]);

  // History stack for Undo
  const historyRef = useRef<number[][]>([]);

  // Hover telemetry
  const [hoverInfo, setHoverInfo] = useState<{
    timeSec: number;
    frame: number;
    depthMm: number;
    thicknessMm: number;
  } | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize Default Synthetic / Template Dataset
  useEffect(() => {
    loadModelMetadata().then((meta) => {
      if (meta) {
        setStatusMessage(`Model AI aktif: ${meta.model_name} v${meta.version} (${meta.architecture})`);
      } else {
        setStatusMessage('Studio siap. Unggah sonogram atau gunakan data template.');
      }
    });

    const times: number[] = [];
    const sup: number[] = [];
    const deep: number[] = [];
    const red: number[] = [];

    // Synthesize 300 frames with 2 contraction peaks (~10 seconds duration)
    for (let i = 0; i < totalDefaultFrames; i++) {
      const t = (i / totalDefaultFrames) * 12.0; // 0 to 12s
      times.push(t);

      // Superficial aponeurosis stays around ~14.0 mm
      const s = 14.0 + Math.sin(t * 0.5) * 0.4;
      sup.push(s);

      // Deep aponeurosis baseline ~24.0 mm
      const d = 24.0 + Math.sin(t * 0.5) * 0.5;
      deep.push(d);

      // Kinetic red zone: baseline ~30.0 mm with two contraction bursts
      const burst1 = Math.exp(-Math.pow((t - 4.0) / 0.6, 2)) * 14.0;
      const burst2 = Math.exp(-Math.pow((t - 8.5) / 0.7, 2)) * 16.0;
      const r = d + 6.0 + burst1 + burst2;
      red.push(r);
    }

    setTimestamps(times);
    setSuperficialMm(sup);
    setDeepMm(deep);
    setRedMm(red);
    historyRef.current = [red];
  }, []);

  // Compute live kinematics and metrics
  const { frames, summary }: { frames: FrameKinematics[]; summary: BiomechanicsSummary } = computeKinematics(
    superficialMm,
    deepMm,
    redMm,
    timestamps
  );

  // Handle Red Curve Modification (Drawing)
  const handleRedCurveChange = useCallback((newRedMm: number[]) => {
    historyRef.current.push([...newRedMm]);
    if (historyRef.current.length > 40) historyRef.current.shift();
    setRedMm(newRedMm);
  }, []);

  // Handle Undo
  const handleUndo = useCallback(() => {
    if (historyRef.current.length > 1) {
      historyRef.current.pop();
      const previous = historyRef.current[historyRef.current.length - 1];
      setRedMm([...previous]);
      setStatusMessage('Undo goresan berhasil dibatalkan.');
    }
  }, []);

  // Keyboard Shortcuts (Ctrl+Z, Ctrl+S)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        handleUndo();
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        handleExportCsv();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo]);

  // Handle File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const src = event.target?.result as string;
      setImageSrc(src);
      setStatusMessage(`Citra berhasil dimuat: ${file.name}`);
    };
    reader.readAsDataURL(file);
  };

  // Run AI Auto-Segmentation
  const handleRunAiSegmentation = async () => {
    if (!canvasRef.current) return;
    setIsAiRunning(true);
    setStatusMessage('Menjalankan inferensi ONNX WebAssembly...');

    try {
      const result = await runSegmentationInference(canvasRef.current, (msg) => {
        setStatusMessage(msg);
      });

      if (result) {
        // Interpolate predicted curves across current timeline
        const n = timestamps.length;
        const newSup: number[] = [];
        const newDeep: number[] = [];
        const newRed: number[] = [];

        for (let i = 0; i < n; i++) {
          const ratio = i / Math.max(1, n - 1);
          const srcIdx = Math.min(result.superficialMm.length - 1, Math.round(ratio * (result.superficialMm.length - 1)));
          newSup.push(result.superficialMm[srcIdx]);
          newDeep.push(result.deepMm[srcIdx]);
          newRed.push(result.deepMm[srcIdx] + 8.0);
        }

        setSuperficialMm(newSup);
        setDeepMm(newDeep);
        setRedMm(newRed);
        historyRef.current.push([...newRed]);
        setStatusMessage('AI Auto-Segmentation selesai! Mask dan kurva berhasil disinkronkan.');
      } else {
        setStatusMessage('Model ONNX lokal belum terinstal. Menggunakan data heuristik aponeurosis.');
      }
    } catch (err) {
      console.error(err);
      setStatusMessage('Gagal menjalankan inferensi AI.');
    } finally {
      setIsAiRunning(false);
    }
  };

  // Export CSV
  const handleExportCsv = () => {
    exportKinematicsToCsv(frames, 'Biceps_Brachii');
    setStatusMessage('Data kinematika frame-by-frame berhasil diekspor ke format CSV.');
  };

  // Export PNG
  const handleExportPng = () => {
    if (canvasRef.current) {
      exportCanvasToPng(canvasRef.current, 'Biceps_Brachii');
      setStatusMessage('Tangkapan kanvas beresolusi tinggi berhasil disimpan sebagai PNG.');
    }
  };

  return (
    <div className="min-h-screen bg-[#0b1120] text-[#f8fafc] flex flex-col justify-between">
      {/* Top Application Area */}
      <div className="p-4 max-w-[1700px] w-full mx-auto space-y-4">
        {/* Header and Action Toolbar */}
        <HeaderToolbar
          toolMode={toolMode}
          onSetToolMode={setToolMode}
          brushRadius={brushRadius}
          onSetBrushRadius={setBrushRadius}
          zoom={zoom}
          onSetZoom={setZoom}
          onUndo={handleUndo}
          onExportCsv={handleExportCsv}
          onExportPng={handleExportPng}
          onRunAiSegmentation={handleRunAiSegmentation}
          onFileUpload={handleFileUpload}
          isAiRunning={isAiRunning}
        />

        {/* Biomechanics Metric Summary Cards */}
        <MetricsCards summary={summary} />

        {/* Interactive Sonogram Canvas */}
        <SonogramCanvas
          canvasRef={canvasRef}
          imageSrc={imageSrc}
          superficialMm={superficialMm}
          deepMm={deepMm}
          redMm={redMm}
          timestamps={timestamps}
          toolMode={toolMode}
          brushRadius={brushRadius}
          zoom={zoom}
          onRedCurveChange={handleRedCurveChange}
          onHoverData={(t, f, d, th) => setHoverInfo({ timeSec: t, frame: f, depthMm: d, thicknessMm: th })}
        />

        {/* Real-time Synchronized Kinematics Chart */}
        <KinematicsChart frames={frames} restBaselineMm={summary.restThicknessMm} />
      </div>

      {/* Global Real-Time Status Bar */}
      <StatusBar statusMessage={statusMessage} hoverInfo={hoverInfo} />
    </div>
  );
}

export default App;
