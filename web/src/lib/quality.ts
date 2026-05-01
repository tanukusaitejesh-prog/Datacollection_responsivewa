import { PoseLandmarkerResult } from '@mediapipe/tasks-vision';

export interface ReadinessCheck {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
}

export interface ReadinessResult {
  ready: boolean;
  summary: string;
  checks: ReadinessCheck[];
  quality: PoseFrameQuality;
}

export interface PoseFrameQuality {
  hasPose: boolean;
  usable: boolean;
  score: number;
  averageVisibility: number;
  reliableLandmarks: number;
  inFrameLandmarks: number;
  coreLandmarks: number;
  bodyBoxArea: number;
  reason: string;
}

const EXPECTED_LANDMARKS = 33;
const TRUSTED_VISIBILITY = 0.2;
const MIN_AVERAGE_VISIBILITY = 0.16;
const MIN_RELIABLE_LANDMARKS = 10;
const MIN_IN_FRAME_LANDMARKS = 12;
const MIN_CORE_LANDMARKS = 3;
const IN_FRAME_TOLERANCE = 0.08;
const CORE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24];

function isFiniteLandmark(lm: any): boolean {
  return Number.isFinite(lm?.x) && Number.isFinite(lm?.y) && Number.isFinite(lm?.z);
}

function isInFrame(lm: any): boolean {
  return (
    isFiniteLandmark(lm) &&
    lm.x >= -IN_FRAME_TOLERANCE &&
    lm.x <= 1 + IN_FRAME_TOLERANCE &&
    lm.y >= -IN_FRAME_TOLERANCE &&
    lm.y <= 1 + IN_FRAME_TOLERANCE
  );
}

function landmarkVisibility(lm: any): number {
  return Number.isFinite(lm?.visibility) ? lm.visibility : 0;
}

export function analyzePoseFrame(results: PoseLandmarkerResult | null): PoseFrameQuality {
  if (!results?.landmarks?.length) {
    return {
      hasPose: false,
      usable: false,
      score: 0,
      averageVisibility: 0,
      reliableLandmarks: 0,
      inFrameLandmarks: 0,
      coreLandmarks: 0,
      bodyBoxArea: 0,
      reason: 'No person detected'
    };
  }

  const landmarks = results.landmarks[0] || [];
  const validLandmarks = landmarks.filter(isFiniteLandmark);
  const reliableLandmarks = validLandmarks.filter(lm => landmarkVisibility(lm) >= TRUSTED_VISIBILITY).length;
  const inFrameLandmarks = validLandmarks.filter(isInFrame).length;
  const coreLandmarks = CORE_LANDMARKS.filter(i => {
    const lm = landmarks[i];
    return isInFrame(lm) && landmarkVisibility(lm) >= TRUSTED_VISIBILITY;
  }).length;

  const averageVisibility =
    validLandmarks.length > 0
      ? validLandmarks.reduce((sum, lm) => sum + landmarkVisibility(lm), 0) / validLandmarks.length
      : 0;

  const boxLandmarks = validLandmarks.filter(isInFrame);
  let bodyBoxArea = 0;
  if (boxLandmarks.length > 1) {
    const xs = boxLandmarks.map(lm => lm.x);
    const ys = boxLandmarks.map(lm => lm.y);
    bodyBoxArea = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
  }

  const coverageScore = Math.min(1, reliableLandmarks / EXPECTED_LANDMARKS);
  const frameScore = Math.min(1, inFrameLandmarks / EXPECTED_LANDMARKS);
  const coreScore = Math.min(1, coreLandmarks / CORE_LANDMARKS.length);
  const visibilityScore = Math.min(1, averageVisibility / 0.5);
  const score = Math.round((coverageScore * 0.35 + frameScore * 0.25 + coreScore * 0.25 + visibilityScore * 0.15) * 100);

  const usable =
    validLandmarks.length === EXPECTED_LANDMARKS &&
    reliableLandmarks >= MIN_RELIABLE_LANDMARKS &&
    inFrameLandmarks >= MIN_IN_FRAME_LANDMARKS &&
    coreLandmarks >= MIN_CORE_LANDMARKS &&
    averageVisibility >= MIN_AVERAGE_VISIBILITY;

  let reason = `Pose tracked (${score}/100)`;
  if (validLandmarks.length !== EXPECTED_LANDMARKS) {
    reason = `Incomplete pose (${validLandmarks.length}/${EXPECTED_LANDMARKS} landmarks)`;
  } else if (reliableLandmarks < MIN_RELIABLE_LANDMARKS) {
    reason = `Low-confidence pose (${reliableLandmarks}/${EXPECTED_LANDMARKS} trusted landmarks)`;
  } else if (inFrameLandmarks < MIN_IN_FRAME_LANDMARKS) {
    reason = `Too many landmarks outside camera view (${inFrameLandmarks}/${EXPECTED_LANDMARKS} in frame)`;
  } else if (coreLandmarks < MIN_CORE_LANDMARKS) {
    reason = 'Head/torso pose is not stable enough';
  } else if (averageVisibility < MIN_AVERAGE_VISIBILITY) {
    reason = 'Pose confidence is too low';
  }

  return {
    hasPose: true,
    usable,
    score,
    averageVisibility,
    reliableLandmarks,
    inFrameLandmarks,
    coreLandmarks,
    bodyBoxArea,
    reason
  };
}

export function isPoseFrameUsable(results: PoseLandmarkerResult | null): boolean {
  return analyzePoseFrame(results).usable;
}

export function evaluateCaptureReadiness(
  results: PoseLandmarkerResult | null,
  detectorReady: boolean
): ReadinessResult {
  const quality = analyzePoseFrame(results);
  const checks: ReadinessCheck[] = [
    {
      id: 'detector',
      label: 'Camera',
      ok: detectorReady,
      detail: detectorReady ? 'Ready' : 'Initializing...'
    }
  ];

  if (!detectorReady || !quality.hasPose) {
    checks.push({
      id: 'pose',
      label: 'Pose Lock',
      ok: false,
      detail: 'No person detected'
    });
    return {
      ready: false,
      summary: detectorReady ? 'Please stand in front of the camera' : 'Initializing AI...',
      checks,
      quality
    };
  }

  checks.push({
    id: 'pose',
    label: 'Pose Lock',
    ok: quality.usable,
    detail: quality.usable ? `Stable (${quality.score}/100)` : quality.reason
  });

  checks.push({
    id: 'coverage',
    label: 'Landmarks',
    ok: quality.inFrameLandmarks >= MIN_IN_FRAME_LANDMARKS && quality.reliableLandmarks >= MIN_RELIABLE_LANDMARKS,
    detail: `${quality.reliableLandmarks}/${EXPECTED_LANDMARKS} trusted, ${quality.inFrameLandmarks}/${EXPECTED_LANDMARKS} in frame`
  });

  const ready = detectorReady && quality.usable;

  return {
    ready,
    summary: ready ? 'Ready to record any action' : 'Waiting for a clearer pose',
    checks,
    quality
  };
}
