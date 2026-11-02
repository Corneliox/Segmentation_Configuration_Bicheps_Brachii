import React from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Repeat,
  Gauge,
  Clock,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { FrameKinematics, AppLanguage } from '../types/ultrasound';
import { TRANSLATIONS } from '../utils/translations';

interface PhotoshopTimelineProps {
  language: AppLanguage;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStepPrev: () => void;
  onStepNext: () => void;
  isLooping: boolean;
  onToggleLoop: () => void;
  fps: number;
  onSetFps: (fps: number) => void;
  currentFrameIdx: number;
  totalFrames: number;
  onSeekFrame: (frameIdx: number) => void;
  frames: FrameKinematics[];
  restBaselineMm: number;
  peakMm: number;
}

export const PhotoshopTimeline: React.FC<PhotoshopTimelineProps> = ({
  language,
  isPlaying,
  onTogglePlay,
  onStepPrev,
  onStepNext,
  isLooping,
  onToggleLoop,
  fps,
  onSetFps,
  currentFrameIdx,
  totalFrames,
  onSeekFrame,
  frames,
  restBaselineMm,
  peakMm,
}) => {
  const t = TRANSLATIONS[language];
  const currentTimeSec = totalFrames > 0 ? (currentFrameIdx / Math.max(1, fps)) : 0;
  const totalTimeSec = totalFrames > 0 ? (totalFrames / Math.max(1, fps)) : 0;

  // Subsample frames if length > 400 for high-performance chart rendering
  const step = Math.max(1, Math.floor(frames.length / 300));
  const chartData = frames
    .filter((_, idx) => idx % step === 0)
    .map((f) => ({
      time: Number(f.timestampSec.toFixed(2)),
      frame: f.frameIndex,
      thickness: Number(f.muscleThicknessMm.toFixed(2)),
    }));

  return (
    <footer className="border-t border-[var(--border-subtle)] bg-[var(--bg-surface)] select-none text-[var(--text-primary)]">
      {/* Top Transport Controls & Timecode Bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border-subtle)]/60 text-xs">
        {/* Left: Media Transport Buttons */}
        <div className="flex items-center gap-2">
          {/* Step Back */}
          <button
            onClick={onStepPrev}
            className="w-7 h-7 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] flex items-center justify-center border border-[var(--border-subtle)] transition"
            title="Previous Frame (Left Arrow)"
          >
            <SkipBack size={13} />
          </button>

          {/* Play / Pause Button */}
          <button
            onClick={onTogglePlay}
            className="px-3 h-7 rounded-lg font-bold flex items-center gap-1.5 transition shadow-sm text-black"
            style={{
              background: 'linear-gradient(135deg, #a8c7fa 0%, #d0bcff 100%)',
            }}
            title="Play / Pause (Space)"
          >
            {isPlaying ? <Pause size={14} className="fill-current" /> : <Play size={14} className="fill-current" />}
            <span className="text-[11px] font-black">{isPlaying ? 'PAUSE' : 'PLAY'}</span>
          </button>

          {/* Step Forward */}
          <button
            onClick={onStepNext}
            className="w-7 h-7 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--bg-card-hover)] flex items-center justify-center border border-[var(--border-subtle)] transition"
            title="Next Frame (Right Arrow)"
          >
            <SkipForward size={13} />
          </button>

          {/* Loop Toggle */}
          <button
            onClick={onToggleLoop}
            className={`w-7 h-7 rounded-lg flex items-center justify-center border transition ${
              isLooping
                ? 'bg-[var(--gemini-purple)]/20 border-[var(--gemini-purple)] text-[var(--gemini-purple)]'
                : 'bg-[var(--bg-card)] border-[var(--border-subtle)] text-[var(--text-muted)] hover:bg-[var(--bg-card-hover)]'
            }`}
            title="Toggle Loop Playback"
          >
            <Repeat size={13} />
          </button>

          {/* Speed / FPS Selector */}
          <div className="flex items-center gap-1 ml-2 bg-[var(--bg-card)] px-2 py-1 rounded-lg border border-[var(--border-subtle)] text-[11px]">
            <Gauge size={12} className="text-[var(--gemini-blue)]" />
            <select
              value={fps}
              onChange={(e) => onSetFps(Number(e.target.value))}
              className="bg-transparent text-[var(--text-primary)] font-bold outline-none cursor-pointer"
            >
              <option value={15} className="bg-[var(--bg-card)]">15 FPS (Slow)</option>
              <option value={30} className="bg-[var(--bg-card)]">30 FPS</option>
              <option value={60} className="bg-[var(--bg-card)]">60 FPS (Standard)</option>
              <option value={112} className="bg-[var(--bg-card)]">112 FPS (Native Cine)</option>
            </select>
          </div>
        </div>

        {/* Center: Interactive Scrubber Slider */}
        <div className="flex-1 max-w-xl mx-4 flex items-center gap-2">
          <input
            type="range"
            min="0"
            max={Math.max(0, totalFrames - 1)}
            value={currentFrameIdx}
            onChange={(e) => onSeekFrame(Number(e.target.value))}
            className="w-full h-1.5 accent-[var(--gemini-blue)] cursor-pointer"
          />
        </div>

        {/* Right: Timecode Display */}
        <div className="flex items-center gap-3 font-mono text-[11px]">
          <div className="flex items-center gap-1.5 bg-[var(--bg-card)] px-2.5 py-1 rounded-lg border border-[var(--border-subtle)]">
            <Clock size={12} className="text-[var(--gemini-yellow)]" />
            <span className="text-[var(--gemini-blue)] font-bold">
              {currentTimeSec.toFixed(2)}s
            </span>
            <span className="text-[var(--text-muted)]">/ {totalTimeSec.toFixed(2)}s</span>
          </div>

          <div className="bg-[var(--bg-card)] px-2.5 py-1 rounded-lg border border-[var(--border-subtle)] font-bold">
            <span className="text-[var(--gemini-emerald)]">Frame {currentFrameIdx + 1}</span>
            <span className="text-[var(--text-muted)]"> / {totalFrames}</span>
          </div>
        </div>
      </div>

      {/* Embedded Synchronized Kinematics Waveform Graph */}
      <div className="px-4 py-2 h-28 w-full bg-[var(--timeline-bg)]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 6, right: 10, left: -20, bottom: 0 }}
            onClick={(e) => {
              if (e && e.activePayload && e.activePayload.length > 0) {
                const targetFrame = e.activePayload[0].payload.frame;
                onSeekFrame(targetFrame);
              }
            }}
          >
            <defs>
              <linearGradient id="geminiKinematicsGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a8c7fa" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#a8c7fa" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} opacity={0.5} />
            <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={10} tickFormatter={(v) => `${v}s`} />
            <YAxis stroke="var(--text-muted)" fontSize={10} domain={['auto', 'auto']} unit="mm" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--bg-card)',
                borderColor: 'var(--border-subtle)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
                fontSize: '11px',
              }}
              formatter={(val: number) => [`${val.toFixed(2)} mm`, 'Thickness']}
              labelFormatter={(label) => `Time: ${label}s`}
            />

            {/* Resting Baseline Reference Line */}
            {restBaselineMm > 0 && (
              <ReferenceLine
                y={restBaselineMm}
                stroke="#fde293"
                strokeDasharray="4 4"
                label={{ value: 'Rest', fill: '#fde293', fontSize: 9, position: 'insideTopLeft' }}
              />
            )}

            {/* Current Playback Scrubber Needle in Chart */}
            <ReferenceLine
              x={Number(currentTimeSec.toFixed(2))}
              stroke="#ffffff"
              strokeWidth={2}
            />

            <Area
              type="monotone"
              dataKey="thickness"
              stroke="#a8c7fa"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#geminiKinematicsGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </footer>
  );
};
