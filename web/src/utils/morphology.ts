/**
 * Morphological Operations & Connected Components Analysis
 * Aligned 1:1 with Python OpenCV pipeline (utils.postprocess_mask)
 */

// Exact OpenCV cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) offsets
const ELLIPSE_5X5_OFFSETS: [number, number][] = [
  // dy, dx
  [-2, 0],
  [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
  [0, -2],  [0, -1],  [0, 0],  [0, 1],  [0, 2],
  [1, -2],  [1, -1],  [1, 0],  [1, 1],  [1, 2],
  [2, 0],
];

/**
 * Morphological dilation with 5x5 elliptical structuring element
 */
export function dilate5x5(src: Uint8Array, width: number, height: number): Uint8Array {
  const dst = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (src[y * width + x] === 0) continue;
      for (let k = 0; k < ELLIPSE_5X5_OFFSETS.length; k++) {
        const ny = y + ELLIPSE_5X5_OFFSETS[k][0];
        const nx = x + ELLIPSE_5X5_OFFSETS[k][1];
        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
          dst[ny * width + nx] = 1;
        }
      }
    }
  }
  return dst;
}

/**
 * Morphological erosion with 5x5 elliptical structuring element
 */
export function erode5x5(src: Uint8Array, width: number, height: number): Uint8Array {
  const dst = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let keep = true;
      for (let k = 0; k < ELLIPSE_5X5_OFFSETS.length; k++) {
        const ny = y + ELLIPSE_5X5_OFFSETS[k][0];
        const nx = x + ELLIPSE_5X5_OFFSETS[k][1];
        if (ny < 0 || ny >= height || nx < 0 || nx >= width || src[ny * width + nx] === 0) {
          keep = false;
          break;
        }
      }
      if (keep) {
        dst[y * width + x] = 1;
      }
    }
  }
  return dst;
}

/**
 * Morphological Closing: Dilation followed by Erosion
 */
export function morphologicalClose(src: Uint8Array, width: number, height: number): Uint8Array {
  const dilated = dilate5x5(src, width, height);
  return erode5x5(dilated, width, height);
}

/**
 * Morphological Opening: Erosion followed by Dilation
 */
export function morphologicalOpen(src: Uint8Array, width: number, height: number): Uint8Array {
  const eroded = erode5x5(src, width, height);
  return dilate5x5(eroded, width, height);
}

/**
 * 8-Connected Component Analysis to extract the single largest foreground component
 * Eliminates background speckle noise and acoustic reverberation artifacts.
 */
export function keepLargestConnectedComponent(
  src: Uint8Array,
  width: number,
  height: number,
  minSize: number = 400
): Uint8Array {
  const labels = new Int32Array(width * height);
  let currentLabel = 0;
  const componentSizes: number[] = [0]; // label 0 = background

  // 8-neighbor connectivity offsets (dy, dx)
  const NEIGHBORS_8: [number, number][] = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1],
  ];

  // Flood-fill BFS to label components
  const queueX = new Int32Array(width * height);
  const queueY = new Int32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (src[idx] === 1 && labels[idx] === 0) {
        currentLabel++;
        let size = 0;

        let head = 0;
        let tail = 0;
        queueX[tail] = x;
        queueY[tail] = y;
        tail++;
        labels[idx] = currentLabel;

        while (head < tail) {
          const cx = queueX[head];
          const cy = queueY[head];
          head++;
          size++;

          for (let k = 0; k < 8; k++) {
            const nx = cx + NEIGHBORS_8[k][1];
            const ny = cy + NEIGHBORS_8[k][0];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const nidx = ny * width + nx;
              if (src[nidx] === 1 && labels[nidx] === 0) {
                labels[nidx] = currentLabel;
                queueX[tail] = nx;
                queueY[tail] = ny;
                tail++;
              }
            }
          }
        }

        componentSizes.push(size);
      }
    }
  }

  const result = new Uint8Array(width * height);
  if (currentLabel === 0) return result;

  // Find label with maximum area among foreground labels (1..currentLabel)
  let maxLabel = 1;
  let maxSize = componentSizes[1];
  for (let l = 2; l <= currentLabel; l++) {
    if (componentSizes[l] > maxSize) {
      maxSize = componentSizes[l];
      maxLabel = l;
    }
  }

  // If even the largest component is smaller than minSize, return empty
  if (maxSize < minSize) {
    return result;
  }

  for (let i = 0; i < width * height; i++) {
    if (labels[i] === maxLabel) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Full postprocessing matching Python utils.postprocess_mask:
 * 1. Threshold at 0.5
 * 2. Morphological Close (ellipse 5x5)
 * 3. Morphological Open (ellipse 5x5)
 * 4. Keep largest 8-connected component
 */
export function postprocessMask(
  rawProb: Float32Array,
  width: number,
  height: number,
  threshold: number = 0.5,
  minSize: number = 400
): Uint8Array {
  const binary = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    if (rawProb[i] >= threshold) {
      binary[i] = 1;
    }
  }

  const closed = morphologicalClose(binary, width, height);
  const opened = morphologicalOpen(closed, width, height);
  return keepLargestConnectedComponent(opened, width, height, minSize);
}

/**
 * Nearest-neighbor resize matching cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
 */
export function resizeMaskNearest(
  srcMask: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const sy = Math.min(srcH - 1, Math.floor(y * scaleY));
    const srcRowOffset = sy * srcW;
    const dstRowOffset = y * dstW;
    for (let x = 0; x < dstW; x++) {
      const sx = Math.min(srcW - 1, Math.floor(x * scaleX));
      dst[dstRowOffset + x] = srcMask[srcRowOffset + sx];
    }
  }

  return dst;
}

/**
 * Telemetry and Thickness extraction matching Python utils.calculate_muscle_thickness
 */
export function calculateMuscleTelemetry(
  maskOrig: Uint8Array,
  origW: number,
  origH: number,
  pixelSpacingMm: number
): {
  supDepthMm: number;
  deepDepthMm: number;
  meanThicknessMm: number;
  colTopsMm: number[];
  colBottomsMm: number[];
  thicknessProfileMm: number[];
} {
  const colTops: number[] = [];
  const colBottoms: number[] = [];
  const thicknessProfile: number[] = [];

  for (let c = 0; c < origW; c++) {
    let topY = -1;
    let bottomY = -1;

    for (let r = 0; r < origH; r++) {
      if (maskOrig[r * origW + c] > 0) {
        if (topY === -1) topY = r;
        bottomY = r;
      }
    }

    if (topY !== -1 && bottomY !== -1) {
      const topMm = topY * pixelSpacingMm;
      const bottomMm = bottomY * pixelSpacingMm;
      colTops.push(topMm);
      colBottoms.push(bottomMm);
      thicknessProfile.push((bottomY - topY) * pixelSpacingMm);
    }
  }

  if (colTops.length === 0) {
    return {
      supDepthMm: 14.0,
      deepDepthMm: 28.0,
      meanThicknessMm: 14.0,
      colTopsMm: [],
      colBottomsMm: [],
      thicknessProfileMm: [],
    };
  }

  const supDepthMm = colTops.reduce((a, b) => a + b, 0) / colTops.length;
  const deepDepthMm = colBottoms.reduce((a, b) => a + b, 0) / colBottoms.length;
  const meanThicknessMm = thicknessProfile.reduce((a, b) => a + b, 0) / thicknessProfile.length;

  return {
    supDepthMm,
    deepDepthMm,
    meanThicknessMm,
    colTopsMm: colTops,
    colBottomsMm: colBottoms,
    thicknessProfileMm: thicknessProfile,
  };
}
