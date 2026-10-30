import { FrameKinematics, BiomechanicsSummary } from '../types/ultrasound';

export function exportKinematicsToCsv(frames: FrameKinematics[], subjectName: string = 'Biceps_Brachii') {
  if (frames.length === 0) return;

  const headers = [
    'Frame_Index',
    'Timestamp_Sec',
    'State',
    'Superficial_Fascia_Depth_mm',
    'Deep_Fascia_Depth_mm',
    'Red_Boundary_Depth_mm',
    'Biceps_Thickness_mm',
  ];

  const rows = frames.map((f) => [
    f.frameIndex + 1,
    f.timestampSec.toFixed(3),
    f.state,
    f.superficialDepthMm.toFixed(3),
    f.deepDepthMm.toFixed(3),
    f.redBoundaryDepthMm.toFixed(3),
    f.muscleThicknessMm.toFixed(3),
  ]);

  const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `${subjectName}_kinematics_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportCanvasToPng(canvas: HTMLCanvasElement, subjectName: string = 'Biceps_Brachii') {
  const url = canvas.toDataURL('image/png');
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `${subjectName}_annotated_sonogram_${Date.now()}.png`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
