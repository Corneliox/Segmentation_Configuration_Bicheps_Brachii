import * as ort from 'onnxruntime-web';
import { ModelMetadata } from '../types/ultrasound';
import {
  postprocessMask,
  resizeMaskNearest,
  calculateMuscleTelemetry,
} from '../utils/morphology';

let inferenceSession: ort.InferenceSession | null = null;
let currentMetadata: ModelMetadata | null = null;

// Initialize ONNX Web runtime
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
ort.env.wasm.simd = true;

export async function loadModelMetadata(): Promise<ModelMetadata | null> {
  try {
    const res = await fetch('./models/model_metadata.json');
    if (!res.ok) return null;
    currentMetadata = await res.json();
    return currentMetadata;
  } catch (err) {
    console.warn('Could not load model_metadata.json:', err);
    return null;
  }
}

export async function getOrLoadSession(onProgress?: (msg: string) => void): Promise<ort.InferenceSession | null> {
  if (inferenceSession) return inferenceSession;

  try {
    if (onProgress) onProgress('Initializing ONNX Runtime Web session...');

    const modelPath = './models/biceps_brachii_unet_quantized.onnx';
    const fallbackPath = './models/biceps_brachii_unet.onnx';

    let selectedPath = modelPath;
    try {
      const check = await fetch(modelPath, { method: 'HEAD' });
      if (!check.ok) selectedPath = fallbackPath;
    } catch {
      selectedPath = fallbackPath;
    }

    if (onProgress) onProgress(`Loading neural weights from ${selectedPath}...`);

    inferenceSession = await ort.InferenceSession.create(selectedPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    if (onProgress) onProgress('AI Model ready for live segmentation.');
    return inferenceSession;
  } catch (err) {
    console.error('Failed to initialize ONNX session:', err);
    return null;
  }
}

/**
 * Helper to load an HTMLImageElement from a URL or data URI
 */
function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = src;
  });
}

export interface FrameSegmentationResult {
  maskOrig: Uint8Array;
  origW: number;
  origH: number;
  supDepthMm: number;
  deepDepthMm: number;
  meanThicknessMm: number;
  colTopsMm: number[];
  colBottomsMm: number[];
  thicknessProfileMm: number[];
}

/**
 * Runs 1:1 Python equivalent segmentation inference on a single raw ultrasound image:
 * 1. Load pure raw image (no UI overlays, badges, or divider lines)
 * 2. Bilinear resize to 224x512
 * 3. Grayscale conversion and normalization: (x / 255.0 - 0.5) / 0.5
 * 4. ONNX forward pass [1, 1, 512, 224]
 * 5. Morphological post-processing (threshold 0.5, closing 5x5, opening 5x5, keep largest connected component)
 * 6. Nearest neighbor resize back to original resolution (origW x origH)
 * 7. Muscle thickness & aponeurosis depth quantification with pixelSpacingMm
 */
export async function runSingleFrameInference(
  imageSource: HTMLImageElement | string,
  pixelSpacingMm: number = 0.1757
): Promise<FrameSegmentationResult | null> {
  const session = await getOrLoadSession();
  if (!session) return null;

  let img: HTMLImageElement;
  if (typeof imageSource === 'string') {
    img = await loadImageElement(imageSource);
  } else {
    img = imageSource;
  }

  const origW = img.naturalWidth || img.width || 219;
  const origH = img.naturalHeight || img.height || 512;

  const targetW = 224;
  const targetH = 512;

  // 1. Bilinear resize to model input dimensions (224, 512)
  const offscreen = document.createElement('canvas');
  offscreen.width = targetW;
  offscreen.height = targetH;
  const ctx = offscreen.getContext('2d');
  if (!ctx) return null;

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, targetW, targetH);

  const imgData = ctx.getImageData(0, 0, targetW, targetH);
  const rgba = imgData.data;

  // 2. Grayscale conversion & normalization matching Python:
  // norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
  const floatArr = new Float32Array(targetH * targetW);
  for (let i = 0; i < targetH * targetW; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    floatArr[i] = (gray / 255.0 - 0.5) / 0.5;
  }

  // 3. ONNX forward pass [1, 1, 512, 224]
  const inputTensor = new ort.Tensor('float32', floatArr, [1, 1, targetH, targetW]);
  const inputName = session.inputNames[0] || 'input';
  const results = await session.run({ [inputName]: inputTensor });
  const outputName = session.outputNames[0] || 'output';
  const predData = results[outputName].data as Float32Array;

  // 4. Morphological Post-Processing matching Python utils.postprocess_mask:
  // Threshold >= 0.5, Morphological Close (5x5), Morphological Open (5x5), Largest Connected Component
  const cleanMask = postprocessMask(predData, targetW, targetH, 0.5, 400);

  // 5. Nearest neighbor resize back to original resolution (origW, origH)
  const maskOrig = resizeMaskNearest(cleanMask, targetW, targetH, origW, origH);

  // 6. Muscle thickness and aponeurosis boundaries matching Python utils.calculate_muscle_thickness
  const telemetry = calculateMuscleTelemetry(maskOrig, origW, origH, pixelSpacingMm);

  return {
    maskOrig,
    origW,
    origH,
    supDepthMm: telemetry.supDepthMm,
    deepDepthMm: telemetry.deepDepthMm,
    meanThicknessMm: telemetry.meanThicknessMm,
    colTopsMm: telemetry.colTopsMm,
    colBottomsMm: telemetry.colBottomsMm,
    thicknessProfileMm: telemetry.thicknessProfileMm,
  };
}

export interface CineSequenceSegmentationResult {
  masks: Uint8Array[];
  superficialMm: number[];
  deepMm: number[];
  thicknessMm: number[];
  origW: number;
  origH: number;
}

/**
 * Batch cine sequence segmentation matching Python's SegmentationWorker thread:
 * Loops through all frames, segmenting each with live progress updates.
 */
export async function runCineSequenceInference(
  frameSources: string[],
  pixelSpacingMm: number = 0.1757,
  onProgress?: (pct: number, msg: string) => void,
  abortSignal?: AbortSignal
): Promise<CineSequenceSegmentationResult | null> {
  const session = await getOrLoadSession();
  if (!session) return null;

  const nFrames = frameSources.length;
  if (nFrames === 0) return null;

  const masks: Uint8Array[] = [];
  const superficialMm: number[] = [];
  const deepMm: number[] = [];
  const thicknessMm: number[] = [];

  let origW = 219;
  let origH = 512;

  if (onProgress) onProgress(10, `Segmenting ${nFrames} frames with pure ONNX U-Net...`);

  for (let idx = 0; idx < nFrames; idx++) {
    if (abortSignal?.aborted) {
      if (onProgress) onProgress(0, 'Segmentation cancelled by user.');
      return null;
    }

    const src = frameSources[idx];
    const res = await runSingleFrameInference(src, pixelSpacingMm);

    if (res) {
      origW = res.origW;
      origH = res.origH;
      masks.push(res.maskOrig);
      superficialMm.push(res.supDepthMm);
      deepMm.push(res.deepDepthMm);
      thicknessMm.push(res.meanThicknessMm);
    } else {
      // Fallback if frame failed
      const prevSup = superficialMm.length > 0 ? superficialMm[superficialMm.length - 1] : 14.0;
      const prevDeep = deepMm.length > 0 ? deepMm[deepMm.length - 1] : 28.0;
      masks.push(new Uint8Array(origW * origH));
      superficialMm.push(prevSup);
      deepMm.push(prevDeep);
      thicknessMm.push(prevDeep - prevSup);
    }

    if (idx % 2 === 0 || idx === nFrames - 1) {
      const pct = 10 + Math.floor(((idx + 1) / nFrames) * 88);
      const curTh = thicknessMm[idx];
      if (onProgress) {
        onProgress(pct, `Segmenting frame ${idx + 1}/${nFrames} (Thick: ${curTh.toFixed(2)} mm)...`);
      }
      // Yield to main thread for responsive UI
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  if (onProgress) onProgress(100, 'AI segmentation complete!');

  return {
    masks,
    superficialMm,
    deepMm,
    thicknessMm,
    origW,
    origH,
  };
}
