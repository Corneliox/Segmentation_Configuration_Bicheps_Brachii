import * as ort from 'onnxruntime-web';
import { ModelMetadata } from '../types/ultrasound';

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

export async function runSegmentationInference(
  canvas: HTMLCanvasElement,
  onProgress?: (msg: string) => void
): Promise<{
  superficialMm: number[];
  deepMm: number[];
  maskData: Uint8Array;
} | null> {
  const session = await getOrLoadSession(onProgress);
  if (!session) return null;

  const targetH = 512;
  const targetW = 224;

  // 1. Resize input canvas to model dimensions (512, 224)
  const offscreen = document.createElement('canvas');
  offscreen.width = targetW;
  offscreen.height = targetH;
  const ctx = offscreen.getContext('2d');
  if (!ctx) return null;

  ctx.drawImage(canvas, 0, 0, targetW, targetH);
  const imgData = ctx.getImageData(0, 0, targetW, targetH);
  const rgba = imgData.data;

  // 2. Preprocess: Convert to grayscale and normalize: (pixel / 255.0 - 0.5) / 0.5
  const floatArr = new Float32Array(targetH * targetW);
  for (let i = 0; i < targetH * targetW; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    floatArr[i] = (gray / 255.0 - 0.5) / 0.5;
  }

  // 3. Create ONNX Tensor [1, 1, 512, 224]
  const inputTensor = new ort.Tensor('float32', floatArr, [1, 1, targetH, targetW]);
  const inputName = session.inputNames[0] || 'input';

  if (onProgress) onProgress('Running neural segmentation inference...');
  const results = await session.run({ [inputName]: inputTensor });
  const outputName = session.outputNames[0] || 'output';
  const outputTensor = results[outputName];
  const predData = outputTensor.data as Float32Array;

  // 4. Post-processing: extract boundary curves and binarize
  const maskData = new Uint8Array(targetH * targetW);
  const superficialMm: number[] = new Array(targetW).fill(14.0);
  const deepMm: number[] = new Array(targetW).fill(28.0);

  const dzMm = 89.96 / targetH; // 0.1757 mm/pixel

  for (let x = 0; x < targetW; x++) {
    let topY = -1;
    let bottomY = -1;

    for (let y = 0; y < targetH; y++) {
      const idx = y * targetW + x;
      const prob = predData[idx];
      if (prob >= 0.5) {
        maskData[idx] = 255;
        if (topY === -1) topY = y;
        bottomY = y;
      }
    }

    if (topY !== -1) {
      superficialMm[x] = topY * dzMm;
      deepMm[x] = bottomY * dzMm;
    }
  }

  return { superficialMm, deepMm, maskData };
}
