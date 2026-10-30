import React from 'react';

interface StatusBarProps {
  statusMessage: string;
  hoverInfo: {
    timeSec: number;
    frame: number;
    depthMm: number;
    thicknessMm: number;
  } | null;
}

export const StatusBar: React.FC<StatusBarProps> = ({ statusMessage, hoverInfo }) => {
  return (
    <div className="bg-[#020617] border-t border-[#1e293b] px-4 py-2 flex items-center justify-between text-xs font-mono text-[#94a3b8]">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-[#4ade80]"></span>
        <span className="text-[#e2e8f0] font-sans">{statusMessage}</span>
      </div>

      {hoverInfo ? (
        <div className="flex items-center gap-4 text-[11px]">
          <span>
            TIME: <strong className="text-white">{hoverInfo.timeSec.toFixed(2)}s</strong>
          </span>
          <span>
            FRAME: <strong className="text-white">{hoverInfo.frame}</strong>
          </span>
          <span>
            DEPTH: <strong className="text-[#38bdf8]">{hoverInfo.depthMm.toFixed(2)} mm</strong>
          </span>
          <span>
            THICKNESS:{' '}
            <strong className="text-[#f97316]">{hoverInfo.thicknessMm.toFixed(2)} mm</strong>
          </span>
        </div>
      ) : (
        <span className="text-[11px] text-[#64748b]">Arahkan kursor pada kanvas untuk melihat metrik titik</span>
      )}
    </div>
  );
};
