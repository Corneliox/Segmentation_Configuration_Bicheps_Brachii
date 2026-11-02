export interface ModelMetadata {
  model_name: string;
  version: string;
  framework: string;
  architecture: string;
  input_tensor: {
    name: string;
    shape: [number, number, number, number];
    channels: number;
    data_type: string;
    color_space: string;
    normalization: {
      scale: number;
      mean: number;
      std: number;
      formula: string;
    };
  };
  output_tensor: {
    name: string;
    shape: [number, number, number, number];
    data_type: string;
    activation: string;
    range: [number, number];
    binary_threshold: number;
  };
  biomechanics: {
    pixel_spacing_mm: number;
    aperture_width_mm: number;
    aperture_depth_mm: number;
    active_contraction_threshold_mm: number;
  };
  artifacts: {
    fp32_model: string;
    quantized_model: string | null;
  };
}

export interface FrameKinematics {
  frameIndex: number;
  timestampSec: number;
  state: 'ACTIVE' | 'RELAX';
  superficialDepthMm: number;
  deepDepthMm: number;
  redBoundaryDepthMm: number;
  muscleThicknessMm: number;
}

export interface BiomechanicsSummary {
  restThicknessMm: number;
  peakThicknessMm: number;
  deltaThicknessMm: number;
  percentageIncrease: number;
  activeFramesCount: number;
  totalFrames: number;
}

export interface UltrasoundPreset {
  id: string;
  name: string;
  probe_model: string;
  probe_type: string;
  frequency_mhz: number;
  frequency_range_mhz?: string;
  element_count: number;
  pitch_mm?: number;
  sound_velocity_m_s: number;
  dynamic_range_db?: number;
  focal_depth_mm?: number;
  max_depth_mm: number;
  native_fps: number;
  pixel_spacing_mm: number;
  target_muscle: string;
  notes?: string;
}

export interface LayerVisibilityState {
  showMask: boolean;
  showSuperficial: boolean;
  showDeep: boolean;
  showGrid: boolean;
  showRawBMode: boolean;
  maskOpacity: number;
}

export type ToolMode = 'pointer' | 'pan' | 'pen' | 'eraser' | 'caliper' | 'zoom';
export type ViewportMode = 'overlay' | 'side_by_side' | 'raw' | 'filmstrip';
export type AppTheme = 'dark' | 'light';
export type AppLanguage = 'en' | 'zh';
