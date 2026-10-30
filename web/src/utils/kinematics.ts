import { FrameKinematics, BiomechanicsSummary } from '../types/ultrasound';

export function computeKinematics(
  superficialMm: number[],
  deepMm: number[],
  redMm: number[],
  timestamps: number[],
  thresholdContractionDeltaMm: number = 2.0
): { frames: FrameKinematics[]; summary: BiomechanicsSummary } {
  const n = timestamps.length;
  if (n === 0) {
    return {
      frames: [],
      summary: {
        restThicknessMm: 0,
        peakThicknessMm: 0,
        deltaThicknessMm: 0,
        percentageIncrease: 0,
        activeFramesCount: 0,
        totalFrames: 0,
      },
    };
  }

  // Raw thickness values: thickness between deep and superficial, or red and superficial
  const thicknessArray: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = superficialMm[i] ?? 12.0;
    const d = redMm[i] ?? deepMm[i] ?? s + 10.0;
    thicknessArray.push(Math.max(0.5, d - s));
  }

  // Baseline resting thickness: median of lowest 25% of values
  const sorted = [...thicknessArray].sort((a, b) => a - b);
  const quartileIdx = Math.max(1, Math.floor(n * 0.25));
  const restSlice = sorted.slice(0, quartileIdx);
  const restThicknessMm = restSlice.reduce((acc, v) => acc + v, 0) / restSlice.length;

  const activeThreshold = restThicknessMm + thresholdContractionDeltaMm;

  let activeCount = 0;
  let peakThicknessMm = 0;

  const frames: FrameKinematics[] = [];
  for (let i = 0; i < n; i++) {
    const th = thicknessArray[i];
    const isActive = th > activeThreshold;
    if (isActive) activeCount++;
    if (th > peakThicknessMm) peakThicknessMm = th;

    frames.push({
      frameIndex: i,
      timestampSec: Number(timestamps[i].toFixed(3)),
      state: isActive ? 'ACTIVE' : 'RELAX',
      superficialDepthMm: Number(superficialMm[i].toFixed(3)),
      deepDepthMm: Number(deepMm[i].toFixed(3)),
      redBoundaryDepthMm: Number(redMm[i].toFixed(3)),
      muscleThicknessMm: Number(th.toFixed(3)),
    });
  }

  const deltaThicknessMm = Math.max(0, peakThicknessMm - restThicknessMm);
  const percentageIncrease = restThicknessMm > 0 ? (deltaThicknessMm / restThicknessMm) * 100 : 0;

  return {
    frames,
    summary: {
      restThicknessMm: Number(restThicknessMm.toFixed(2)),
      peakThicknessMm: Number(peakThicknessMm.toFixed(2)),
      deltaThicknessMm: Number(deltaThicknessMm.toFixed(2)),
      percentageIncrease: Number(percentageIncrease.toFixed(1)),
      activeFramesCount: activeCount,
      totalFrames: n,
    },
  };
}
