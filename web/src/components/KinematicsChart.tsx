import React from 'react';
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
import { FrameKinematics } from '../types/ultrasound';

interface KinematicsChartProps {
  frames: FrameKinematics[];
  restBaselineMm: number;
}

export const KinematicsChart: React.FC<KinematicsChartProps> = ({ frames, restBaselineMm }) => {
  if (frames.length === 0) return null;

  // Subsample frames if length > 600 for high-FPS chart rendering
  const step = Math.max(1, Math.floor(frames.length / 500));
  const data = frames
    .filter((_, idx) => idx % step === 0)
    .map((f) => ({
      time: f.timestampSec,
      thickness: f.muscleThicknessMm,
      state: f.state,
    }));

  return (
    <div className="bg-[#0f172a] border border-[#334155] rounded-xl p-4 shadow-xl">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-bold text-[#38bdf8] uppercase tracking-wider flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#f97316] animate-pulse"></span>
          Biceps Brachii Dynamic Thickness Waveform (Kinematics)
        </h3>
        <span className="text-xs text-[#94a3b8] font-mono">
          Resting Baseline: <strong className="text-[#fde047]">{restBaselineMm.toFixed(2)} mm</strong>
        </span>
      </div>

      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="thicknessGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f97316" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#f97316" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="time"
              stroke="#64748b"
              fontSize={11}
              tickFormatter={(v) => `${v}s`}
            />
            <YAxis
              stroke="#64748b"
              fontSize={11}
              domain={['auto', 'auto']}
              unit="mm"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0f172a',
                borderColor: '#334155',
                borderRadius: '8px',
                color: '#f8fafc',
                fontSize: '12px',
              }}
              formatter={(val: number) => [`${val.toFixed(2)} mm`, 'Thickness']}
              labelFormatter={(label) => `Time: ${label}s`}
            />

            <ReferenceLine
              y={restBaselineMm}
              stroke="#fde047"
              strokeDasharray="4 4"
              label={{ value: 'Rest', fill: '#fde047', fontSize: 10, position: 'insideTopLeft' }}
            />

            <Area
              type="monotone"
              dataKey="thickness"
              stroke="#f97316"
              strokeWidth={2.2}
              fillOpacity={1}
              fill="url(#thicknessGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
