export const DEFAULT_PIXEL_SPACING_MM = 0.1757; // 1 pixel = 0.1757 mm
export const DEFAULT_DEPTH_MAX_MM = 89.96;

export function mmToPixel(depthMm: number, canvasHeight: number, maxDepthMm: number = DEFAULT_DEPTH_MAX_MM): number {
  return Math.min(canvasHeight, Math.max(0, (depthMm / maxDepthMm) * canvasHeight));
}

export function pixelToMm(pixelY: number, canvasHeight: number, maxDepthMm: number = DEFAULT_DEPTH_MAX_MM): number {
  return (Math.max(0, Math.min(canvasHeight, pixelY)) / canvasHeight) * maxDepthMm;
}

export function frameToPixelX(frameIndex: number, totalFrames: number, canvasWidth: number): number {
  if (totalFrames <= 1) return 0;
  return (frameIndex / (totalFrames - 1)) * canvasWidth;
}

export function pixelXToFrame(pixelX: number, totalFrames: number, canvasWidth: number): number {
  if (totalFrames <= 1 || canvasWidth <= 0) return 0;
  const ratio = Math.max(0, Math.min(1, pixelX / canvasWidth));
  return Math.round(ratio * (totalFrames - 1));
}
