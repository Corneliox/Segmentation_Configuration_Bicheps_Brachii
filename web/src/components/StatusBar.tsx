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
    <div className="bg-[var(--bg-surface)] border-t border-[var(--border-subtle)] px-4 py-1.5 flex items-center justify-between text-xs font-mono text-[var(--text-secondary)] select-none">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-[var(--gemini-emerald)] animate-pulse" />
        <span className="text-[var(--text-primary)] font-sans text-[11px]">{statusMessage}</span>
      </div>

      {hoverInfo ? (
        <div className="flex items-center gap-4 text-[11px]">
          <span>
            TIME: <strong className="text-[var(--text-primary)]">{hoverInfo.timeSec.toFixed(2)}s</strong>
          </span>
          <span>
            FRAME: <strong className="text-[var(--text-primary)]">{hoverInfo.frame}</strong>
          </span>
          <span>
            DEPTH: <strong className="text-[var(--gemini-blue)]">{hoverInfo.depthMm.toFixed(2)} mm</strong>
          </span>
          <span>
            THICKNESS:{' '}
            <strong className="text-[var(--gemini-purple)]">{hoverInfo.thicknessMm.toFixed(2)} mm</strong>
          </span>
        </div>
      ) : (
        <span className="text-[11px] text-[var(--text-muted)]">
          Hover over canvas or timeline to view pinpoint spatial telemetry
        </span>
      )}
    </div>
  );
};
