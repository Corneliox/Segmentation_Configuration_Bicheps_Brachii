import { FrameKinematics, BiomechanicsSummary } from '../types/ultrasound';

/**
 * Calculates percentile from an array of numbers
 */
function getPercentile(arr: number[], percentile: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

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

  // Thickness per frame: deep fascia minus superficial fascia (or red curve if modified)
  const thicknessArray: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = superficialMm[i] ?? 14.0;
    const d = redMm[i] ?? deepMm[i] ?? 28.0;
    thicknessArray.push(Math.max(0.1, d - s));
  }

  // Baseline resting thickness: 10th percentile matching Python np.percentile(thickness_mm, 10)
  const restThicknessMm = getPercentile(thicknessArray, 10);
  const peakThicknessMm = Math.max(...thicknessArray);
  const deltaThicknessMm = Math.max(0, peakThicknessMm - restThicknessMm);
  const percentageIncrease = restThicknessMm > 0 ? (deltaThicknessMm / restThicknessMm) * 100 : 0;

  // Active contraction detection (active if thickness > rest + delta threshold or > 18.0 mm)
  const activeThreshold = Math.max(18.0, restThicknessMm + thresholdContractionDeltaMm);

  let activeCount = 0;
  const frames: FrameKinematics[] = [];
  for (let i = 0; i < n; i++) {
    const th = thicknessArray[i];
    const isActive = th > activeThreshold;
    if (isActive) activeCount++;

    frames.push({
      frameIndex: i,
      timestampSec: Number(timestamps[i].toFixed(3)),
      state: isActive ? 'ACTIVE' : 'RELAX',
      superficialDepthMm: Number(superficialMm[i].toFixed(3)),
      deepDepthMm: Number(deepMm[i].toFixed(3)),
      redBoundaryDepthMm: Number((redMm[i] ?? deepMm[i]).toFixed(3)),
      muscleThicknessMm: Number(th.toFixed(3)),
    });
  }

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
