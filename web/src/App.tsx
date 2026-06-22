import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { PoseEngine } from './lib/pose-engine';
import { 
  Square, 
  AlertCircle, 
  RefreshCw,
  ArrowLeft,
  ChevronRight,
  UploadCloud,
  FlipHorizontal,
  FileCheck,
  CheckCircle2,
  XCircle
} from 'lucide-react';
import { PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import { supabase } from './lib/supabase';
import { analyzePoseFrame, evaluateCaptureReadiness, type PoseFrameQuality } from './lib/quality';
import { validatePoseData, ValidationResult, createNpyBuffer, normalizePoseSequence, resampleSequence30FPS, resampleBlendshapes30FPS } from './lib/validator-utils';
import { Validator } from './Validator';

const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'N/A' },
] as const;

const CENTER_OPTIONS = [
  { value: 'barkatpura', label: 'Barkatpura' },
  { value: 'neredmet', label: 'Neredmet' },
  { value: 'other', label: 'Other' },
] as const;

type CenterName = (typeof CENTER_OPTIONS)[number]['value'];

/*
const LANDMARK_NAMES = [
  'NOSE', 'L_EYE_I', 'L_EYE', 'L_EYE_O', 'R_EYE_I', 'R_EYE', 'R_EYE_O', 
  'L_EAR', 'R_EAR', 'MOUTH_L', 'MOUTH_R', 'L_SHOU', 'R_SHOU', 'L_ELBO', 
  'R_ELBO', 'L_WRIS', 'R_WRIS', 'L_PINK', 'R_PINK', 'L_INDE', 'R_INDE', 
  'L_THUM', 'R_THUM', 'L_HIP', 'R_HIP', 'L_KNEE', 'R_KNEE', 'L_ANKL', 
  'R_ANKL', 'L_HEEL', 'R_HEEL', 'L_FOOT', 'R_FOOT'
];
*/

type Gender = (typeof GENDER_OPTIONS)[number]['value'];
type Step = 'home' | 'testing' | 'recording' | 'confirm' | 'validator';
type CameraFacing = 'user' | 'environment';
type CaptureMode = 'pose_only' | 'holistic' | 'half_body';

type RecordedPoseQuality = Pick<PoseFrameQuality, 'score' | 'averageVisibility' | 'reliableLandmarks' | 'inFrameLandmarks' | 'bodyBoxArea'> & {
  timestampMs: number;
};

const FACE_LANDMARK_COUNT = 478;
const HAND_LANDMARK_COUNT = 42;
const POSE_LANDMARK_COUNT = 33;

function coordOrNaN(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : Number.NaN;
}

function missingPoint(): [number, number, number] {
  return [Number.NaN, Number.NaN, Number.NaN];
}

function missingLandmarkFrame(count: number): [number, number, number][] {
  return Array.from({ length: count }, missingPoint);
}

function missingLandmarkFrames(frameCount: number, landmarkCount: number): [number, number, number][][] {
  return Array.from({ length: frameCount }, () => missingLandmarkFrame(landmarkCount));
}

function missingBlendshapeFrames(frameCount: number): number[][] {
  return Array.from({ length: frameCount }, () => [Number.NaN]);
}

function stringifyWithNaN(value: unknown): string {
  return JSON.stringify(value, (_key, item) => (
    typeof item === 'number' && Number.isNaN(item) ? 'NaN' : item
  ));
}

function average(values: number[]): number | null {
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function minimum(values: number[]): number | null {
  return values.length > 0 ? Math.min(...values) : null;
}

function roundMetric(value: number | null, digits = 3): number | null {
  return value === null ? null : Number(value.toFixed(digits));
}

function summarizePoseQuality(samples: RecordedPoseQuality[], skippedFrames: number) {
  const scores = samples.map(sample => sample.score);
  const visibility = samples.map(sample => sample.averageVisibility);
  const reliable = samples.map(sample => sample.reliableLandmarks);
  const inFrame = samples.map(sample => sample.inFrameLandmarks);
  const boxAreas = samples.map(sample => sample.bodyBoxArea);

  return {
    accepted_frames: samples.length,
    skipped_frames: skippedFrames,
    mean_score: roundMetric(average(scores), 1),
    min_score: roundMetric(minimum(scores), 1),
    mean_visibility: roundMetric(average(visibility)),
    min_visibility: roundMetric(minimum(visibility)),
    mean_reliable_landmarks: roundMetric(average(reliable), 1),
    min_reliable_landmarks: roundMetric(minimum(reliable), 1),
    mean_in_frame_landmarks: roundMetric(average(inFrame), 1),
    min_in_frame_landmarks: roundMetric(minimum(inFrame), 1),
    mean_body_box_area: roundMetric(average(boxAreas), 4),
    min_body_box_area: roundMetric(minimum(boxAreas), 4)
  };
}



function generateClinicalCsvString(frames: number[][][], timestamps: number[]): string {
  if (frames.length === 0) return '';

  const KINECT_JOINT_ORDER = [
    'Midspain', 'AnkleLeft', 'AnkleRight', 'ElbowLeft', 'ElbowRight',
    'FootLeft', 'FootRight', 'HandLeft', 'HandRight', 'HandTipLeft',
    'HandTipRight', 'Head', 'HipLeft', 'HipRight', 'KneeLeft',
    'KneeRight', 'Neck', 'ShoulderLeft', 'ShoulderRight', 'SpineBase',
    'SpineShoulder', 'ThumbLeft', 'ThumbRight', 'WristLeft', 'WristRight'
  ];

  const formatTimestamp = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const milliseconds = Math.floor(ms % 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}:${milliseconds.toString().padStart(3, '0')})`;
  };

  const headers = ['H:M:S:MS)'];
  KINECT_JOINT_ORDER.forEach(joint => {
    headers.push(`${joint}-x`, `${joint}-y`, `${joint}-z`);
  });

  const csvRows = [headers.join(',')];
  const grid: Record<string, number[]>[] = [];

  frames.forEach((frame) => {
    const getXYZ = (idx: number) => {
      const p = frame[idx];
      if (!p || (p[0] === 0 && p[1] === 0 && p[2] === 0) || !Number.isFinite(p[0])) {
        return [NaN, NaN, NaN];
      }
      return [p[0] * -1.0, p[1] * -1.0, p[2]];
    };

    const getMid = (idx1: number, idx2: number) => {
      const p1 = frame[idx1];
      const p2 = frame[idx2];
      if (!p1 || !p2 || 
          (p1[0] === 0 && p1[1] === 0 && p1[2] === 0) || 
          (p2[0] === 0 && p2[1] === 0 && p2[2] === 0) ||
          !Number.isFinite(p1[0]) || !Number.isFinite(p2[0])) {
        return [NaN, NaN, NaN];
      }
      return [
        ((p1[0] + p2[0]) / 2.0) * -1.0,
        ((p1[1] + p2[1]) / 2.0) * -1.0,
        (p1[2] + p2[2]) / 2.0
      ];
    };

    const spineShoulder = getMid(11, 12);
    const spineBase = getMid(23, 24);
    let midspain = [NaN, NaN, NaN];
    if (Number.isFinite(spineShoulder[0]) && Number.isFinite(spineBase[0])) {
      midspain = [
        (spineShoulder[0] + spineBase[0]) / 2.0,
        (spineShoulder[1] + spineBase[1]) / 2.0,
        (spineShoulder[2] + spineBase[2]) / 2.0
      ];
    }

    grid.push({
      'Midspain': midspain,
      'AnkleLeft': getXYZ(27),
      'AnkleRight': getXYZ(28),
      'ElbowLeft': getXYZ(13),
      'ElbowRight': getXYZ(14),
      'FootLeft': getXYZ(31),
      'FootRight': getXYZ(32),
      'HandLeft': getXYZ(15),
      'HandRight': getXYZ(16),
      'HandTipLeft': getXYZ(19),
      'HandTipRight': getXYZ(20),
      'Head': getXYZ(0),
      'HipLeft': getXYZ(23),
      'HipRight': getXYZ(24),
      'KneeLeft': getXYZ(25),
      'KneeRight': getXYZ(26),
      'Neck': getMid(11, 12),
      'ShoulderLeft': getXYZ(11),
      'ShoulderRight': getXYZ(12),
      'SpineBase': spineBase,
      'SpineShoulder': spineShoulder,
      'ThumbLeft': getXYZ(21),
      'ThumbRight': getXYZ(22),
      'WristLeft': getXYZ(15),
      'WristRight': getXYZ(16)
    });
  });

  const T = frames.length;

  // Linear interpolation matching pandas df.interpolate(method='linear', limit_direction='both')
  // plus ffill() and bfill() and fallback to 0.0
  KINECT_JOINT_ORDER.forEach(joint => {
    for (let axis = 0; axis < 3; axis++) {
      const validIndices: number[] = [];
      for (let t = 0; t < T; t++) {
        if (Number.isFinite(grid[t][joint][axis])) {
          validIndices.push(t);
        }
      }

      if (validIndices.length === 0) {
        for (let t = 0; t < T; t++) {
          grid[t][joint][axis] = 0.0;
        }
      } else {
        for (let t = 0; t < T; t++) {
          if (!Number.isFinite(grid[t][joint][axis])) {
            if (t < validIndices[0]) {
              grid[t][joint][axis] = grid[validIndices[0]][joint][axis];
            } else if (t > validIndices[validIndices.length - 1]) {
              grid[t][joint][axis] = grid[validIndices[validIndices.length - 1]][joint][axis];
            } else {
              let prevIdx = validIndices[0];
              let nextIdx = validIndices[validIndices.length - 1];
              for (let i = 0; i < validIndices.length; i++) {
                if (validIndices[i] < t) prevIdx = validIndices[i];
                if (validIndices[i] > t) {
                  nextIdx = validIndices[i];
                  break;
                }
              }
              const fraction = (t - prevIdx) / (nextIdx - prevIdx);
              const valPrev = grid[prevIdx][joint][axis];
              const valNext = grid[nextIdx][joint][axis];
              grid[t][joint][axis] = valPrev + (valNext - valPrev) * fraction;
            }
          }
        }
      }
    }
  });

  // Construct CSV String
  frames.forEach((_, t) => {
    const ts = timestamps[t] || 0;
    const timeStr = formatTimestamp(ts);
    const row: string[] = [timeStr];

    KINECT_JOINT_ORDER.forEach(joint => {
      const coords = grid[t][joint];
      row.push(coords[0].toFixed(6), coords[1].toFixed(6), coords[2].toFixed(6));
    });

    csvRows.push(row.join(','));
  });

  return csvRows.join('\n');
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function uploadToSupabaseDirect(
  payload: any,
  captureId: string,
  csvString: string,
  npyPoseOnlyBuffer: ArrayBuffer,
  npyPoseHandsBuffer: ArrayBuffer
) {
  try {
    const slugify = (t: string) => (t || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)+/g, '');
    const slugSubject = slugify(payload.meta?.subject_name) || 'subject';
    
    const jsonFileName = `${captureId}/${slugSubject}_raw_capture.json`;
    const npyFileName = `${captureId}/${slugSubject}_keypoints.npy`;
    const npyPoseHandsFileName = `${captureId}/${slugSubject}_keypoints_pose_hands.npy`;
    const csvFileName = `${captureId}/${slugSubject}_keypoints.csv`;
    
    const jsonBlob = new Blob([stringifyWithNaN(payload)], { type: 'application/json' });
    const npyBlob = new Blob([npyPoseOnlyBuffer], { type: 'application/octet-stream' });
    const npyPoseHandsBlob = new Blob([npyPoseHandsBuffer], { type: 'application/octet-stream' });
    const csvBlob = new Blob([csvString], { type: 'text/csv' });

    // 1. Upload JSON
    const { error: jsonError } = await supabase.storage
      .from('pose-captures')
      .upload(jsonFileName, jsonBlob, { upsert: true });

    if (jsonError) throw jsonError;

    // 2. Upload Pose-only NPY
    const { error: npyError } = await supabase.storage
      .from('pose-captures')
      .upload(npyFileName, npyBlob, { upsert: true });

    if (npyError) throw npyError;

    // 3. Upload Pose+Hands NPY
    const { error: npyPoseHandsError } = await supabase.storage
      .from('pose-captures')
      .upload(npyPoseHandsFileName, npyPoseHandsBlob, { upsert: true });

    if (npyPoseHandsError) throw npyPoseHandsError;

    // 4. Upload CSV
    const { error: csvError } = await supabase.storage
      .from('pose-captures')
      .upload(csvFileName, csvBlob, { upsert: true });

    if (csvError) throw csvError;

    // 5. Insert DB record
    const { error: dbError } = await supabase.from('captures').insert({
      capture_id: captureId,
      meta: payload.meta,
      storage_paths: {
        raw_json: jsonFileName,
        keypoints_npy: npyFileName,
        keypoints_pose_hands_npy: npyPoseHandsFileName,
        keypoints_csv: csvFileName
      }
    });

    if (dbError) throw dbError;
    return true;
  } catch (err) {
    console.error('Web Supabase direct sync failed:', err);
    return false;
  }
}

async function pushToMongoDirect(
  payload: any,
  captureId: string,
  npyPoseOnlyBuffer: ArrayBuffer,
  npyPoseHandsBuffer: ArrayBuffer
) {
  const FUNCTION_URL = '/.netlify/functions/pushToMongo';

  async function sendChunk(body: any, label: string) {
    const response = await fetch(FUNCTION_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: stringifyWithNaN(body),
    });
    if (!response.ok) {
      let errMsg = `${label} failed (HTTP ${response.status})`;
      try {
        const errData = await response.json();
        errMsg = errData.error || errData.message || errMsg;
      } catch {
        const text = await response.text();
        if (text.includes('<!DOCTYPE') || text.includes('<html')) {
          errMsg = `${label}: payload too large or function error (${response.status})`;
        }
      }
      throw new Error(errMsg);
    }
    return true;
  }

  try {
    // Chunk 1: metadata + keypoints (no binary)
    console.log('[MongoDB] Uploading metadata...');
    await sendChunk({ captureId, chunk: 'meta', ...payload }, 'Metadata upload');

    // Chunk 2: NPY pose-only file
    console.log('[MongoDB] Uploading NPY pose file...');
    const npyBase64 = arrayBufferToBase64(npyPoseOnlyBuffer);
    await sendChunk({ captureId, chunk: 'npy_pose', npy_file: npyBase64 }, 'NPY pose upload');

    // Chunk 3: NPY pose+hands file
    console.log('[MongoDB] Uploading NPY pose+hands file...');
    const npyPoseHandsBase64 = arrayBufferToBase64(npyPoseHandsBuffer);
    await sendChunk({ captureId, chunk: 'npy_hands', npy_file_pose_hands: npyPoseHandsBase64 }, 'NPY hands upload');

    console.log('[MongoDB] All chunks uploaded successfully.');
    return true;
  } catch (err: any) {
    console.error('Web MongoDB direct push failed:', err);
    return { success: false, error: err.message };
  }
}


export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<PoseEngine | null>(null);
  
  // Recording Data
  const framesRef = useRef<number[][][]>([]);
  const faceFramesRef = useRef<number[][][]>([]);
  const handFramesRef = useRef<number[][][]>([]);
  const faceBlendshapesRef = useRef<any[]>([]);
  const timestampsRef = useRef<number[]>([]);
  const poseQualityRef = useRef<RecordedPoseQuality[]>([]);
  const skippedFramesRef = useRef(0);
  const cameraResolutionRef = useRef<[number, number]>([1280, 720]);
  const startTimeRef = useRef<number>(0);
  const isRecordingRef = useRef(false);
  const captureModeRef = useRef<CaptureMode>('holistic');
  const faceDetectedFramesRef = useRef(0);
  const handDetectedFramesRef = useRef(0);

  // Warnings State and Refs
  const [multiplePeopleWarning, setMultiplePeopleWarning] = useState(false);
  const [occlusionWarnings, setOcclusionWarnings] = useState<string[]>([]);
  
  const multiplePeopleDetectedDuringRecording = useRef(false);
  const occludedLimbsDuringRecording = useRef<string[]>([]);
  const lowVisibilityCountsRef = useRef<Record<number, number>>({
    25: 0, 26: 0, 27: 0, 28: 0, 31: 0, 32: 0
  });

  // App State
  const [step, setStep] = useState<Step>('home');
  const [isReady, setIsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [skippedFrames, setSkippedFrames] = useState(0);
  const [faceDetectedFrames, setFaceDetectedFrames] = useState(0);
  const [handDetectedFrames, setHandDetectedFrames] = useState(0);
  const previousFramesRef = useRef<number[][] | null>(null); // For smoothing

  const [duration, setDuration] = useState(0);
  const [latestResults, setLatestResults] = useState<PoseLandmarkerResult | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [cameraAspect, setCameraAspect] = useState('16 / 9');

  // Metadata State
  const [sessionId, setSessionId] = useState('');
  const [subjectName, setSubjectName] = useState('');
  const [subjectId, setSubjectId] = useState('');
  const [actionType, setActionType] = useState('');
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<Gender>('prefer_not_to_say');
  const [centerName, setCenterName] = useState<CenterName>(CENTER_OPTIONS[0].value);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>('user');
  const [captureMode, setCaptureMode] = useState<CaptureMode>('holistic');
  const [clinicianNotes, setClinicianNotes] = useState('');
  const [showNotes, setShowNotes] = useState(false);
  const [showPermissionGuide, setShowPermissionGuide] = useState(false);
  const cameraActive = step === 'testing' || step === 'recording';

  useEffect(() => {
    captureModeRef.current = captureMode;
  }, [captureMode]);

  useEffect(() => {
    if (cameraActive) {
      initCamera();
    }
    return () => {
      stopCamera();
    };
  }, [cameraActive, cameraFacing]);

  const stopCamera = () => {
    isRecordingRef.current = false;
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    engineRef.current?.close();
    engineRef.current = null;
    setIsReady(false);
    setLatestResults(null);
    setMultiplePeopleWarning(false);
    setOcclusionWarnings([]);
  };

  const initCamera = async () => {
    try {
      stopCamera();
      await new Promise(r => setTimeout(r, 200));
      if (!videoRef.current || !canvasRef.current) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: cameraFacing,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false,
      });
      
      videoRef.current.srcObject = stream;
      
      const engine = new PoseEngine(videoRef.current, canvasRef.current);
      engine.onResults(handlePoseResults);
      engineRef.current = engine;

      videoRef.current.onloadedmetadata = () => {
        const videoWidth = videoRef.current?.videoWidth || 1280;
        const videoHeight = videoRef.current?.videoHeight || 720;
        cameraResolutionRef.current = [videoWidth, videoHeight];
        setCameraAspect(`${videoWidth} / ${videoHeight}`);
        if (canvasRef.current) {
          canvasRef.current.width = videoWidth;
          canvasRef.current.height = videoHeight;
        }
        setIsReady(true);
        startEngineLoop();
      };
    } catch (err) {
      console.error(err);
      if (window.location.hostname !== 'localhost' && window.location.protocol !== 'https:') {
        setShowPermissionGuide(true);
      } else {
        alert('Camera access failed. Check browser permissions.');
      }
    }
  };

  const startEngineLoop = async () => {
    let isProcessing = false;
    const loop = async () => {
      if (videoRef.current && engineRef.current && !isProcessing) {
        isProcessing = true;
        const v = videoRef.current;
        const c = canvasRef.current;
        
        // Sync canvas resolution
        if (c && v.videoWidth > 0 && (c.width !== v.videoWidth || c.height !== v.videoHeight)) {
          c.width = v.videoWidth;
          c.height = v.videoHeight;
        }

        const useHolistic = captureModeRef.current === 'holistic';
        
        // Run inference in a non-blocking way
        engineRef.current.send(v, {
          face: useHolistic,
          hands: useHolistic
        }).finally(() => {
          isProcessing = false;
        });
      }
      requestAnimationFrame(loop);
    };
    loop();
  };

  const handlePoseResults = (results: any) => {
    setLatestResults(results.pose);
    const poseQuality = analyzePoseFrame(results.pose);

    // 1. Multi-Person Detection Safeguard
    const numPeople = results.pose?.landmarks?.length || 0;
    const isMultiPerson = numPeople > 1;
    setMultiplePeopleWarning(isMultiPerson);
    if (isMultiPerson && isRecordingRef.current) {
      multiplePeopleDetectedDuringRecording.current = true;
    }

    // 2. Real-Time Pose Defect and Occlusion Warnings
    if (results.pose?.landmarks?.length > 0) {
      const landmarks = results.pose.landmarks[0];
      const jointsToTrack = [
        { index: 25, name: 'Left Knee' },
        { index: 26, name: 'Right Knee' },
        { index: 27, name: 'Left Ankle' },
        { index: 28, name: 'Right Ankle' },
        { index: 31, name: 'Left Foot' },
        { index: 32, name: 'Right Foot' }
      ];

      const newOcclusionWarnings: string[] = [];
      let leftFootOccluded = false;
      let rightFootOccluded = false;

      jointsToTrack.forEach(joint => {
        const lm = landmarks[joint.index];
        const vis = lm ? (lm.visibility ?? 0) : 0;
        
        if (vis < 0.45) {
          lowVisibilityCountsRef.current[joint.index] = (lowVisibilityCountsRef.current[joint.index] || 0) + 1;
        } else {
          lowVisibilityCountsRef.current[joint.index] = 0;
        }

        if (lowVisibilityCountsRef.current[joint.index] > 5) {
          if (joint.index === 25) newOcclusionWarnings.push('Left Knee tracking lost.');
          if (joint.index === 26) newOcclusionWarnings.push('Right Knee tracking lost.');
          if (joint.index === 27 || joint.index === 31) leftFootOccluded = true;
          if (joint.index === 28 || joint.index === 32) rightFootOccluded = true;
          
          if (isRecordingRef.current && !occludedLimbsDuringRecording.current.includes(joint.name)) {
            occludedLimbsDuringRecording.current.push(joint.name);
          }
        }
      });

      if (leftFootOccluded && rightFootOccluded) {
        newOcclusionWarnings.push('Lower body occluded. Ensure feet are visible!');
      } else if (leftFootOccluded) {
        newOcclusionWarnings.push('Left Foot/Ankle occluded.');
      } else if (rightFootOccluded) {
        newOcclusionWarnings.push('Right Foot/Ankle occluded.');
      }

      setOcclusionWarnings(newOcclusionWarnings);
    } else {
      setOcclusionWarnings([]);
    }

    if (isRecordingRef.current) {
      const elapsed = Date.now() - startTimeRef.current;
      
      if (!poseQuality.usable || !results.pose?.landmarks?.length) {
        skippedFramesRef.current += 1;
        setSkippedFrames(skippedFramesRef.current);
        
        framesRef.current.push(missingLandmarkFrame(POSE_LANDMARK_COUNT).map(p => [...p, 0])); 
        poseQualityRef.current.push({
          score: 0,
          averageVisibility: 0,
          reliableLandmarks: 0,
          inFrameLandmarks: 0,
          bodyBoxArea: 0,
          timestampMs: elapsed
        });
        
        faceFramesRef.current.push(missingLandmarkFrame(FACE_LANDMARK_COUNT));
        faceBlendshapesRef.current.push([Number.NaN]);
        handFramesRef.current.push(missingLandmarkFrame(HAND_LANDMARK_COUNT));
        
        timestampsRef.current.push(elapsed);
        setFrameCount(framesRef.current.length);
        setDuration(elapsed);
        return;
      }

      // Use worldLandmarks (meters) instead of landmarks (normalized) for training consistency
      const poseLandmarks = results.pose.worldLandmarks?.[0] || results.pose.landmarks[0];
      const frame = poseLandmarks.map((lm: any) => [
        coordOrNaN(lm.x), coordOrNaN(lm.y), coordOrNaN(lm.z), coordOrNaN(lm.visibility || 0)
      ]);

      
      // Apply Temporal Smoothing (EMA) if in Half Body mode to prevent flickering
      if (captureModeRef.current === 'half_body' && previousFramesRef.current) {
        const smoothing = 0.65; // Higher = more weight to new frame, lower = more smoothing
        frame.forEach((lm: number[], i: number) => {
          const prev = previousFramesRef.current![i];
          if (prev && !lm.some(isNaN) && !prev.some(isNaN)) {

            lm[0] = prev[0] * (1 - smoothing) + lm[0] * smoothing;
            lm[1] = prev[1] * (1 - smoothing) + lm[1] * smoothing;
            lm[2] = prev[2] * (1 - smoothing) + lm[2] * smoothing;
            // Don't smooth visibility as much
            lm[3] = prev[3] * 0.3 + lm[3] * 0.7;
          }
        });
      }

      // In half_body mode, we aggressively hide low-confidence lower body landmarks
      if (captureModeRef.current === 'half_body') {
        frame.forEach((lm: number[], i: number) => {

          // Landmarks 25-32 are knees, ankles, heels, feet
          if (i >= 25 && lm[3] < 0.45) {
            lm[0] = 0; lm[1] = 0; lm[2] = 0; lm[3] = 0;
          }
        });
      }
      
      previousFramesRef.current = frame.map((f: number[]) => [...f]);

      framesRef.current.push(frame);

      poseQualityRef.current.push({
        score: poseQuality.score,
        averageVisibility: poseQuality.averageVisibility,
        reliableLandmarks: poseQuality.reliableLandmarks,
        inFrameLandmarks: poseQuality.inFrameLandmarks,
        bodyBoxArea: poseQuality.bodyBoxArea,
        timestampMs: elapsed
      });

      if (results.face && results.face.faceLandmarks && results.face.faceLandmarks.length > 0) {
        faceFramesRef.current.push(results.face.faceLandmarks[0].map((lm: any) => [coordOrNaN(lm.x), coordOrNaN(lm.y), coordOrNaN(lm.z)]));
        faceBlendshapesRef.current.push(results.face.faceBlendshapes?.[0] || []);
        faceDetectedFramesRef.current += 1;
        setFaceDetectedFrames(faceDetectedFramesRef.current);
      } else {
        faceFramesRef.current.push(missingLandmarkFrame(FACE_LANDMARK_COUNT));
        faceBlendshapesRef.current.push([Number.NaN]);
      }

      const rawPoseLandmarks = results.pose?.landmarks?.[0] || [];
      const poseWorldLandmarks = results.pose?.worldLandmarks?.[0] || [];
      const leftHand: [number, number, number][] = Array.from({ length: 21 }, () => [Number.NaN, Number.NaN, Number.NaN]);
      const rightHand: [number, number, number][] = Array.from({ length: 21 }, () => [Number.NaN, Number.NaN, Number.NaN]);
      let handTrackedThisFrame = false;

      // Calculate ratio of world meters to image coordinates for hand scaling
      let metersPerUnit = 1.5;
      if (poseWorldLandmarks.length > 24 && rawPoseLandmarks.length > 24) {
        const L_HIP = 23;
        const R_HIP = 24;
        const L_SHOULDER = 11;
        const R_SHOULDER = 12;

        const wMidHipX = (poseWorldLandmarks[L_HIP].x + poseWorldLandmarks[R_HIP].x) / 2;
        const wMidHipY = (poseWorldLandmarks[L_HIP].y + poseWorldLandmarks[R_HIP].y) / 2;
        const wMidHipZ = (poseWorldLandmarks[L_HIP].z + poseWorldLandmarks[R_HIP].z) / 2;

        const wMidShuX = (poseWorldLandmarks[L_SHOULDER].x + poseWorldLandmarks[R_SHOULDER].x) / 2;
        const wMidShuY = (poseWorldLandmarks[L_SHOULDER].y + poseWorldLandmarks[R_SHOULDER].y) / 2;
        const wMidShuZ = (poseWorldLandmarks[L_SHOULDER].z + poseWorldLandmarks[R_SHOULDER].z) / 2;

        const wTorso = Math.sqrt(
          Math.pow(wMidShuX - wMidHipX, 2) +
          Math.pow(wMidShuY - wMidHipY, 2) +
          Math.pow(wMidShuZ - wMidHipZ, 2)
        );

        const iMidHipX = (rawPoseLandmarks[L_HIP].x + rawPoseLandmarks[R_HIP].x) / 2;
        const iMidHipY = (rawPoseLandmarks[L_HIP].y + rawPoseLandmarks[R_HIP].y) / 2;

        const iMidShuX = (rawPoseLandmarks[L_SHOULDER].x + rawPoseLandmarks[R_SHOULDER].x) / 2;
        const iMidShuY = (rawPoseLandmarks[L_SHOULDER].y + rawPoseLandmarks[R_SHOULDER].y) / 2;

        const iTorso = Math.sqrt(
          Math.pow(iMidShuX - iMidHipX, 2) +
          Math.pow(iMidShuY - iMidHipY, 2)
        );

        if (iTorso > 0.01 && wTorso > 0.01) {
          metersPerUnit = wTorso / iTorso;
        }
      }

      if (results.hands && results.hands.landmarks && results.hands.landmarks.length > 0) {
        results.hands.landmarks.forEach((hand: any, idx: number) => {
          const handedness = results.hands.handedness?.[idx]?.[0]?.categoryName; // "Left" or "Right"
          const targetHand = handedness === 'Left' ? leftHand : rightHand;
          handTrackedThisFrame = true;

          // Find the wrist in world coordinates (from pose landmark 15 or 16)
          const wristPoseIdx = handedness === 'Left' ? 15 : 16;
          const wWrist = poseWorldLandmarks[wristPoseIdx];
          const wWristX = wWrist ? coordOrNaN(wWrist.x) : 0;
          const wWristY = wWrist ? coordOrNaN(wWrist.y) : 0;
          const wWristZ = wWrist ? coordOrNaN(wWrist.z) : 0;

          // Hand landmarks are relative to hand wrist (landmark 0)
          const hWrist = hand[0];
          const hWristX = hWrist ? coordOrNaN(hWrist.x) : 0;
          const hWristY = hWrist ? coordOrNaN(hWrist.y) : 0;
          const hWristZ = hWrist ? coordOrNaN(hWrist.z) : 0;

          hand.forEach((lm: any, i: number) => {
            if (i < 21) {
              const dx = coordOrNaN(lm.x) - hWristX;
              const dy = coordOrNaN(lm.y) - hWristY;
              const dz = coordOrNaN(lm.z) - hWristZ;

              // Convert normalized image offset to meters and anchor to pose wrist
              targetHand[i] = [
                wWristX + dx * metersPerUnit,
                wWristY + dy * metersPerUnit,
                wWristZ + dz * metersPerUnit
              ];
            }
          });
        });
      }

      // Hybrid Fallback: Populate missing hand joints using Pose Landmarker's wrist and fingers in world coordinates
      const pWristL = poseWorldLandmarks[15];
      const pThumbL = poseWorldLandmarks[21];
      const pIndexL = poseWorldLandmarks[19];
      const pPinkyL = poseWorldLandmarks[17];

      if (Number.isNaN(leftHand[0][0]) && pWristL) leftHand[0] = [coordOrNaN(pWristL.x), coordOrNaN(pWristL.y), coordOrNaN(pWristL.z)];
      if (Number.isNaN(leftHand[4][0]) && pThumbL) leftHand[4] = [coordOrNaN(pThumbL.x), coordOrNaN(pThumbL.y), coordOrNaN(pThumbL.z)];
      if (Number.isNaN(leftHand[8][0]) && pIndexL) leftHand[8] = [coordOrNaN(pIndexL.x), coordOrNaN(pIndexL.y), coordOrNaN(pIndexL.z)];
      if (Number.isNaN(leftHand[20][0]) && pPinkyL) leftHand[20] = [coordOrNaN(pPinkyL.x), coordOrNaN(pPinkyL.y), coordOrNaN(pPinkyL.z)];

      const pWristR = poseWorldLandmarks[16];
      const pThumbR = poseWorldLandmarks[22];
      const pIndexR = poseWorldLandmarks[20];
      const pPinkyR = poseWorldLandmarks[18];

      if (Number.isNaN(rightHand[0][0]) && pWristR) rightHand[0] = [coordOrNaN(pWristR.x), coordOrNaN(pWristR.y), coordOrNaN(pWristR.z)];
      if (Number.isNaN(rightHand[4][0]) && pThumbR) rightHand[4] = [coordOrNaN(pThumbR.x), coordOrNaN(pThumbR.y), coordOrNaN(pThumbR.z)];
      if (Number.isNaN(rightHand[8][0]) && pIndexR) rightHand[8] = [coordOrNaN(pIndexR.x), coordOrNaN(pIndexR.y), coordOrNaN(pIndexR.z)];
      if (Number.isNaN(rightHand[20][0]) && pPinkyR) rightHand[20] = [coordOrNaN(pPinkyR.x), coordOrNaN(pPinkyR.y), coordOrNaN(pPinkyR.z)];

      // Combine left and right hand points (total 42 landmarks)
      const combinedHandLandmarks = [...leftHand, ...rightHand];
      handFramesRef.current.push(combinedHandLandmarks);

      if (handTrackedThisFrame) {
        handDetectedFramesRef.current += 1;
        setHandDetectedFrames(handDetectedFramesRef.current);
      }

      timestampsRef.current.push(elapsed);
      setFrameCount(framesRef.current.length);
      setDuration(elapsed);
    }
  };

  const startRecording = () => {
    framesRef.current = [];
    faceFramesRef.current = [];
    handFramesRef.current = [];
    faceBlendshapesRef.current = [];
    timestampsRef.current = [];
    poseQualityRef.current = [];
    skippedFramesRef.current = 0;
    faceDetectedFramesRef.current = 0;
    handDetectedFramesRef.current = 0;
    startTimeRef.current = Date.now();
    setFrameCount(0);
    setSkippedFrames(0);
    setFaceDetectedFrames(0);
    setHandDetectedFrames(0);
    setDuration(0);
    setValidationResult(null);
    previousFramesRef.current = null;
    isRecordingRef.current = true;

    // Reset warnings
    multiplePeopleDetectedDuringRecording.current = false;
    occludedLimbsDuringRecording.current = [];
    lowVisibilityCountsRef.current = { 25: 0, 26: 0, 27: 0, 28: 0, 31: 0, 32: 0 };
    setMultiplePeopleWarning(false);
    setOcclusionWarnings([]);

    setIsRecording(true);
    setStep('recording');
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    setIsRecording(false);
    const resampledFrames = resampleSequence30FPS(framesRef.current, timestampsRef.current);
    const result = validatePoseData(resampledFrames);
    setValidationResult(result);
    setStep('confirm');
  };

  const handleFinalize = async (destination: 'supabase' | 'mongo' | 'both') => {
    setIsUploading(true);
    
    const slugify = (t: string) => (t || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)+/g, '');
    const folderParts = [centerName, subjectName, actionType, age, subjectId].map(slugify).filter(Boolean);
    const folderPrefix = folderParts.join('_');
    const captureId = `${folderPrefix || 'capture'}_${Date.now()}`;
    const safeAge = parseInt(age);
    
    // Resample all sequences to a constant 30 FPS grid
    const resampledFrames = resampleSequence30FPS(framesRef.current, timestampsRef.current);
    const resampledHandFrames = resampleSequence30FPS(handFramesRef.current, timestampsRef.current);
    const resampledFaceFrames = resampleSequence30FPS(faceFramesRef.current, timestampsRef.current);
    const resampledBlendshapes = resampleBlendshapes30FPS(faceBlendshapesRef.current, timestampsRef.current);
    const resampledTimestamps = Array.from({ length: resampledFrames.length }, (_, idx) => idx * (1000 / 30));

    const actualFps = 30.0;
    const recordedFrames = resampledFrames.length;
    const normalizedFrames = normalizePoseSequence(resampledFrames);
    const csvString = generateClinicalCsvString(normalizedFrames, resampledTimestamps);

    const npyPoseOnlyBuffer = createNpyBuffer(normalizedFrames);

    // Create the pose+hands combined raw data (75 landmarks)
    const posePlusHandsRaw = resampledFrames.map((poseFrame, t) => {
      const handFrame = resampledHandFrames[t] || [];
      const handPart = handFrame.map(hlm => [hlm[0], hlm[1], hlm[2], 1.0]);
      return [...poseFrame, ...handPart];
    });
    const normalizedPoseHands = normalizePoseSequence(posePlusHandsRaw);
    const npyPoseHandsBuffer = createNpyBuffer(normalizedPoseHands);

    const payload = {
      keypoints: normalizedFrames,
      csv_data: csvString,
      timestamps: resampledTimestamps,
      face_keypoints: captureMode === 'holistic' ? resampledFaceFrames : missingLandmarkFrames(recordedFrames, FACE_LANDMARK_COUNT),
      face_blendshapes: captureMode === 'holistic' ? resampledBlendshapes : missingBlendshapeFrames(recordedFrames),
      hand_keypoints: captureMode === 'holistic' ? resampledHandFrames : missingLandmarkFrames(recordedFrames, HAND_LANDMARK_COUNT),
      quality: {
        pose: summarizePoseQuality(poseQualityRef.current, skippedFramesRef.current),
        holistic: {
          face_detected_frames: faceDetectedFramesRef.current,
          hand_detected_frames: handDetectedFramesRef.current
        },
        validation: validationResult,
        warnings: {
          multiple_people_detected: multiplePeopleDetectedDuringRecording.current,
          occluded_limbs: occludedLimbsDuringRecording.current
        }
      },
      meta: {
        capture_mode: captureMode,
        fps_nominal: isFinite(actualFps) ? actualFps : 30,
        resolution: cameraResolutionRef.current,
        device: 'Web Chrome',
        camera_facing: cameraFacing,
        session_id: sessionId || `web_${Date.now()}`,
        subject_name: subjectName,
        subject_id: subjectId,
        action_type: actionType,
        age: isNaN(safeAge) ? null : safeAge,
        gender: gender,
        center_name: centerName,
        clinician_notes: clinicianNotes
      }
    };

    let success = false;
    let errorMessage = '';

    if (destination === 'supabase') {
      success = await uploadToSupabaseDirect(payload, captureId, csvString, npyPoseOnlyBuffer, npyPoseHandsBuffer);
      if (!success) errorMessage = 'Unknown Supabase error';
    } else if (destination === 'mongo') {
      const result = await pushToMongoDirect(payload, captureId, npyPoseOnlyBuffer, npyPoseHandsBuffer);
      if (result === true) {
        success = true;
      } else {
        success = false;
        errorMessage = (result as any).error || 'Unknown MongoDB error';
      }
    } else if (destination === 'both') {
      const [sRes, mRes] = await Promise.all([
        uploadToSupabaseDirect(payload, captureId, csvString, npyPoseOnlyBuffer, npyPoseHandsBuffer),
        pushToMongoDirect(payload, captureId, npyPoseOnlyBuffer, npyPoseHandsBuffer)
      ]);
      
      const sSuccess = sRes === true;
      const mSuccess = mRes === true;
      
      if (sSuccess && mSuccess) {
        success = true;
      } else {
        success = false;
        if (!sSuccess) errorMessage += 'Supabase failed. ';
        if (!mSuccess) errorMessage += `MongoDB failed: ${(mRes as any)?.error || 'Unknown'}`;
      }
    }
    
    setIsUploading(false);

    if (success) {
      let destName = '';
      if (destination === 'supabase') destName = 'Cloud';
      else if (destination === 'mongo') destName = 'MongoDB';
      else destName = 'Both Platforms';
      
      alert(`Upload to ${destName} Successful!`);
      setStep('home');
    } else {
      let destName = '';
      if (destination === 'supabase') destName = 'Cloud';
      else if (destination === 'mongo') destName = 'MongoDB';
      else destName = 'Both Platforms';

      alert(`Upload to ${destName} Failed.\nError: ${errorMessage}`);
    }
  };

  const testMongoConnection = async () => {
    setIsUploading(true);
    try {
      const res = await fetch('/.netlify/functions/pushToMongo');
      if (!res.ok) {
        const text = await res.text();
        if (text.trim().startsWith('<!DOCTYPE html>')) {
          alert('MongoDB Connection Failed!\nReason: Netlify functions are not available in the current environment. Please run with "netlify dev".');
        } else {
          alert(`MongoDB Connection Failed!\nStatus: ${res.status}\nReason: ${text.slice(0, 100)}`);
        }
        setIsUploading(false);
        return;
      }
      const data = await res.json();
      if (res.ok) {
        alert('MongoDB Connection Successful!');
      } else {
        alert(`MongoDB Connection Failed!\nReason: ${data.error || data.message}`);
      }
    } catch (err: any) {
      let message = err.message;
      if (message.includes('Unexpected token') || message.includes('DOCTYPE')) {
        message = 'Netlify function not found. Use "netlify dev" to run locally.';
      }
      alert(`Network Error: ${message}`);
    }
    setIsUploading(false);
  };

  const toggleCamera = () => {
    setCameraFacing(prev => prev === 'user' ? 'environment' : 'user');
  };

  const downloadCsv = () => {
    const resampledFrames = resampleSequence30FPS(framesRef.current, timestampsRef.current);
    const resampledTimestamps = Array.from({ length: resampledFrames.length }, (_, idx) => idx * (1000 / 30));
    const normalizedFrames = normalizePoseSequence(resampledFrames);
    const csvString = generateClinicalCsvString(normalizedFrames, resampledTimestamps);
    if (!csvString) return;

    const blob = new Blob([csvString], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `capture_${subjectName || 'unnamed'}_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const readiness = evaluateCaptureReadiness(latestResults, isReady);
  const uploadBlocked = validationResult?.overall === 'fail';
  const cameraSurfaceStyle = { '--camera-aspect': cameraAspect } as CSSProperties;

  return (
    <div className="app-root">
      {step === 'home' && (
        <div className="app-container">
          <div className="step-container">
            <h1 className="home-title">Pose Capture Studio</h1>
            <p className="home-subtitle">Configure subject details to begin.</p>

            <div className="card">
              <div className="card-title">Subject Profile</div>
              <div className="form-grid">
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Full Name</label>
                  <input className="input-field" value={subjectName} onChange={e => setSubjectName(e.target.value)} placeholder="Name" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Subject ID</label>
                  <input className="input-field" value={subjectId} onChange={e => setSubjectId(e.target.value)} placeholder="ID" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Center Name</label>
                  <div className="chip-grid" style={{marginTop: 4}}>
                    {CENTER_OPTIONS.map(opt => (
                      <div 
                        key={opt.value}
                        className={`chip ${centerName === opt.value ? 'active' : ''}`}
                        onClick={() => setCenterName(opt.value)}
                        style={{padding: '6px 12px', fontSize: 11}}
                      >
                        {opt.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{marginTop: 12, marginBottom: 12}}>
                <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Category</label>
                <div className="chip-grid">
                  <div 
                    className={`chip ${actionType === 'asd' ? 'active' : ''}`}
                    onClick={() => setActionType('asd')}
                  >
                    ASD
                  </div>
                  <div 
                    className={`chip ${actionType === 'td' ? 'active' : ''}`}
                    onClick={() => setActionType('td')}
                  >
                    TD
                  </div>
                  <div 
                    className={`chip ${['asd', 'td'].includes(actionType.toLowerCase()) ? '' : actionType ? 'active' : ''}`}
                    onClick={() => setActionType('')}
                  >
                    Other
                  </div>
                </div>
              </div>

              <div className="form-grid form-grid-spaced">
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Age</label>
                  <input type="number" className="input-field" value={age} onChange={e => setAge(e.target.value)} placeholder="Age" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Action Type / Label</label>
                  <input className="input-field" value={actionType} onChange={e => setActionType(e.target.value)} placeholder="e.g. asd, td, or custom action" />
                </div>
              </div>

              <div style={{marginTop: 12}}>
                <button 
                  className="btn btn-secondary" 
                  style={{fontSize: 12, padding: '8px 12px', width: 'auto', marginBottom: showNotes ? 8 : 0}}
                  onClick={() => setShowNotes(!showNotes)}
                >
                  {showNotes ? 'Hide Notes' : 'Add Clinician Notes'}
                </button>
                {showNotes && (
                  <textarea 
                    className="input-field" 
                    style={{minHeight: 80, resize: 'vertical'}}
                    value={clinicianNotes}
                    onChange={e => setClinicianNotes(e.target.value)}
                    placeholder="Enter clinical observations or session notes..."
                  />
                )}
              </div>

              <div style={{marginTop: 12}}>
                <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Gender</label>
                <div className="chip-grid">
                  {GENDER_OPTIONS.map(opt => (
                    <div 
                      key={opt.value}
                      className={`chip ${gender === opt.value ? 'active' : ''}`}
                      onClick={() => setGender(opt.value)}
                    >
                      {opt.label}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-title">Session Settings</div>
              <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Custom Session ID (Optional)</label>
              <input className="input-field" value={sessionId} onChange={e => setSessionId(e.target.value)} placeholder="Auto-generated if empty" />
              
              <div style={{marginTop: 12}}>
                <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Capture Mode</label>
                <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 4}}>
                  <button 
                    className={`chip ${captureMode === 'pose_only' ? 'active' : ''}`}
                    style={{textAlign: 'center'}}
                    onClick={() => setCaptureMode('pose_only')}
                  >
                    Pose
                  </button>
                  <button 
                    className={`chip ${captureMode === 'holistic' ? 'active' : ''}`}
                    style={{textAlign: 'center'}}
                    onClick={() => setCaptureMode('holistic')}
                  >
                    Holistic
                  </button>
                  <button 
                    className={`chip ${captureMode === 'half_body' ? 'active' : ''}`}
                    style={{textAlign: 'center'}}
                    onClick={() => setCaptureMode('half_body')}
                  >
                    Half-Body
                  </button>
                </div>

              </div>

              <button 
                className="btn btn-secondary" 
                style={{marginTop: 15, width: '100%', fontSize: 12, padding: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6}}
                onClick={testMongoConnection}
                disabled={isUploading}
              >
                {isUploading ? <RefreshCw size={14} className="animate-spin" /> : <AlertCircle size={14} />}
                Test MongoDB Connection
              </button>
            </div>

            <button 
              className="btn btn-primary" 
              disabled={!age || !subjectName}
              onClick={() => setStep('testing')}
            >
              Next: Camera Check <ChevronRight size={18} />
            </button>

            <button 
              className="btn btn-secondary" 
              style={{ marginTop: 10 }}
              onClick={() => setStep('validator')}
            >
              <FileCheck size={18} /> Open Data Validator
            </button>
          </div>
        </div>
      )}

      {step === 'validator' && (
        <Validator onBack={() => setStep('home')} />
      )}

      {cameraActive && (
        <div className={`app-container ${isRecording ? 'capture-app-container' : ''}`}>
          <div className={`step-container ${isRecording ? 'capture-step-container' : ''}`}>
            <div className={`camera-wrapper ${isRecording ? 'recording-camera-wrapper' : ''}`} style={cameraSurfaceStyle}>
              <video 
                ref={videoRef} 
                className={`camera-stream ${cameraFacing === 'environment' ? 'back' : ''}`} 
                autoPlay playsInline muted 
              />
              <canvas 
                ref={canvasRef} 
                className={`landmark-canvas ${cameraFacing === 'environment' ? 'back' : ''}`} 
                width={1280} height={720} 
              />
              <button className="flip-btn" onClick={toggleCamera}>
                <FlipHorizontal size={20} />
              </button>

              {/* Warnings Overlay */}
              {(multiplePeopleWarning || occlusionWarnings.length > 0) && (
                <div className="warnings-overlay" style={{
                  position: 'absolute',
                  top: '12px',
                  left: '12px',
                  right: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px',
                  zIndex: 10,
                  pointerEvents: 'none'
                }}>
                  {multiplePeopleWarning && (
                    <div className="warning-banner" style={{
                      backgroundColor: 'rgba(239, 68, 68, 0.95)',
                      color: 'white',
                      padding: '10px 14px',
                      borderRadius: '10px',
                      fontSize: '13px',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.25)',
                      backdropFilter: 'blur(4px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      <AlertCircle size={16} />
                      <span>Multiple people detected in frame. Only the subject should be in camera view.</span>
                    </div>
                  )}
                  {occlusionWarnings.map((warnMsg, idx) => (
                    <div key={idx} className="warning-banner" style={{
                      backgroundColor: 'rgba(217, 119, 6, 0.95)',
                      color: 'white',
                      padding: '10px 14px',
                      borderRadius: '10px',
                      fontSize: '13px',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      boxShadow: '0 4px 15px rgba(0, 0, 0, 0.25)',
                      backdropFilter: 'blur(4px)',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}>
                      <AlertCircle size={16} />
                      <span>{warnMsg}</span>
                    </div>
                  ))}
                </div>
              )}

              {isRecording && (
                <>
                  <div className="recording-bar">
                    REC {(duration/1000).toFixed(1)}s | {frameCount} Frames{captureMode === 'holistic' ? ` | Face ${faceDetectedFrames} | Hands ${handDetectedFrames}` : ''}{skippedFrames > 0 ? ` | ${skippedFrames} skipped` : ''}
                  </div>

                  <div className="stop-btn-overlay">
                    <button className="btn btn-danger" onClick={stopRecording}>
                      <Square size={20} fill="currentColor" /> Stop Recording
                    </button>
                  </div>
                </>
              )}
            </div>

            {!isRecording && (
              <>
                <div className="testing-panel">
                  <div className="card-title">Pre-Capture Check</div>
                  {readiness.checks.map(c => (
                    <p key={c.id} className={`check-item ${c.ok ? 'check-ok' : 'check-fail'}`}>
                      {c.ok ? 'OK' : 'WAIT'} {c.label}: {c.detail}
                    </p>
                  ))}
                  <p style={{marginTop: 8, fontWeight: 700, fontSize: 13, color: readiness.ready ? 'var(--success)' : 'var(--error)'}}>
                    {readiness.summary}
                  </p>
                </div>

                <div className="button-row">
                  <button className="btn btn-secondary" onClick={() => setStep('home')}>
                    <ArrowLeft size={18} /> Back
                  </button>
                  <button className="btn btn-primary" disabled={!readiness.ready} onClick={startRecording}>
                    Start Recording
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {step === 'confirm' && (
        <div className="app-container">
          <div className="step-container">
            <h1 className="home-title">Review & Edit</h1>
            <p className="home-subtitle">Verify details before cloud upload.</p>

            <div className="card">
              <div className="card-title">Captured Data</div>
              <p className="check-item">Frames: {frameCount} ({ (frameCount/(duration/1000 || 1)).toFixed(1) } FPS)</p>
              <p className="check-item">Duration: {(duration/1000).toFixed(1)}s</p>
              {skippedFrames > 0 && (
                <p className="check-item">Skipped low-confidence frames: {skippedFrames}</p>
              )}
              {captureMode === 'holistic' && (
                <p className="check-item">Holistic frames: face {faceDetectedFrames}/{frameCount}, hands {handDetectedFrames}/{frameCount}</p>
              )}
              
              {validationResult && (
                <div style={{ 
                  marginTop: 12, 
                  padding: '10px 12px', 
                  borderRadius: 10, 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 10,
                  backgroundColor: validationResult.overall === 'pass' ? 'rgba(31, 138, 109, 0.1)' : validationResult.overall === 'warn' ? 'rgba(217, 119, 6, 0.1)' : 'rgba(183, 78, 99, 0.1)',
                  border: `1px solid ${validationResult.overall === 'pass' ? 'var(--success)' : validationResult.overall === 'warn' ? '#d97706' : 'var(--error)'}`
                }}>
                  {validationResult.overall === 'pass' ? <CheckCircle2 size={18} color="var(--success)" /> : validationResult.overall === 'warn' ? <AlertCircle size={18} color="#d97706" /> : <XCircle size={18} color="var(--error)" />}
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: validationResult.overall === 'pass' ? 'var(--success)' : validationResult.overall === 'warn' ? '#d97706' : 'var(--error)' }}>
                      {validationResult.overall === 'pass' ? 'Training Ready' : validationResult.overall === 'warn' ? 'Usable (Warnings)' : 'Quality Issue'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                      {validationResult.overall === 'pass' ? 'Meets capture quality checks.' : validationResult.overall === 'warn' ? `${validationResult.warns} warnings detected.` : `${validationResult.fails} critical errors. Record again before uploading.`}
                    </div>
                  </div>
                </div>
              )}
              {uploadBlocked && (
                <p className="check-item check-fail" style={{ marginTop: 10 }}>
                  Upload is disabled because this capture failed quality checks.
                </p>
              )}

              {/* Warnings Summary Card */}
              {(multiplePeopleDetectedDuringRecording.current || occludedLimbsDuringRecording.current.length > 0) && (
                <div style={{
                  marginTop: 12,
                  padding: '10px 12px',
                  borderRadius: 10,
                  backgroundColor: 'rgba(217, 119, 6, 0.1)',
                  border: '1px solid #d97706',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6
                }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#d97706', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <AlertCircle size={16} color="#d97706" /> Capture Warnings
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {multiplePeopleDetectedDuringRecording.current && (
                      <p style={{ margin: '2px 0' }}>• Multiple people were detected in the camera view during this recording.</p>
                    )}
                    {occludedLimbsDuringRecording.current.length > 0 && (
                      <p style={{ margin: '2px 0' }}>• Occluded or low-confidence tracking detected on: {occludedLimbsDuringRecording.current.join(', ')}.</p>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-title">Edit Metadata</div>
              
              <div className="form-grid">
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Name</label>
                  <input className="input-field" value={subjectName} onChange={e => setSubjectName(e.target.value)} />
                </div>
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Subject ID</label>
                  <input className="input-field" value={subjectId} onChange={e => setSubjectId(e.target.value)} />
                </div>
              </div>

              <div className="form-grid form-grid-compact">
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Age</label>
                  <input type="number" className="input-field" value={age} onChange={e => setAge(e.target.value)} />
                </div>
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Action</label>
                  <input className="input-field" value={actionType} onChange={e => setActionType(e.target.value)} />
                </div>
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Center Name</label>
                  <div className="chip-grid" style={{marginTop: 2}}>
                    {CENTER_OPTIONS.map(opt => (
                      <div 
                        key={opt.value}
                        className={`chip ${centerName === opt.value ? 'active' : ''}`}
                        onClick={() => setCenterName(opt.value)}
                        style={{padding: '4px 10px', fontSize: 10}}
                      >
                        {opt.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{marginTop: 10}}>
                <label style={{fontSize: 10, fontWeight: 700}}>Clinician Notes</label>
                <textarea 
                  className="input-field" 
                  style={{minHeight: 60, resize: 'vertical', fontSize: 13}}
                  value={clinicianNotes}
                  onChange={e => setClinicianNotes(e.target.value)}
                  placeholder="Final session notes..."
                />
              </div>
            </div>

            <div style={{marginTop: 'auto', paddingBottom: 20}}>
              <div style={{display: 'flex', flexDirection: 'column', gap: 10}}>
                <button 
                  className="btn btn-primary" 
                  disabled={isUploading || uploadBlocked}
                  onClick={() => handleFinalize('supabase')}
                >
                  {isUploading ? <RefreshCw className="animate-spin" /> : <UploadCloud />}
                  {isUploading ? 'Finalizing...' : 'Finalize & Send to Cloud'}
                </button>
                <button 
                  className="btn btn-primary" 
                  style={{ backgroundColor: '#47A248' }} // MongoDB green color
                  disabled={isUploading || uploadBlocked}
                  onClick={() => handleFinalize('mongo')}
                >
                  {isUploading ? <RefreshCw className="animate-spin" /> : <UploadCloud />}
                  {isUploading ? 'Finalizing...' : 'Finalize & Send to MongoDB'}
                </button>
                <button 
                  className="btn btn-primary" 
                  style={{ 
                    background: 'linear-gradient(135deg, var(--accent) 0%, #47A248 100%)',
                    boxShadow: '0 4px 15px rgba(14, 106, 168, 0.2)'
                  }}
                  disabled={isUploading || uploadBlocked}
                  onClick={() => handleFinalize('both')}
                >
                  {isUploading ? <RefreshCw className="animate-spin" /> : <RefreshCw />}
                  {isUploading ? 'Syncing Both...' : 'Finalize & Sync to Both'}
                </button>

                <button 
                  className="btn btn-secondary" 
                  style={{ backgroundColor: '#f3f4f6', color: '#111827', border: '1px solid #d1d5db', marginTop: 10, width: '100%' }}
                  onClick={downloadCsv}
                >
                  <UploadCloud style={{ transform: 'rotate(180deg)' }} /> Download Exactly as Spreadsheet (CSV)
                </button>
              </div>
              <button 
                className="btn btn-secondary" 
                style={{marginTop: 10, border: 'none', width: '100%'}}
                disabled={isUploading}
                onClick={() => setStep('home')}
              >
                Discard & Start New
              </button>
            </div>
          </div>
        </div>
      )}

      {showPermissionGuide && (
        <div className="modal-overlay">
          <AlertCircle size={64} color="#ff4d4d" style={{ marginBottom: 20 }} />
          <h2>Camera Blocked</h2>
          <p>Chrome blocks cameras on insecure links. To fix:</p>
          <div className="modal-code">
            <p>1. Go to: <b>chrome://flags</b></p>
            <p>2. Enable: <b>Insecure origins treated as secure</b></p>
            <p>3. Add: <b>{window.location.origin}</b></p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowPermissionGuide(false)}>Try Again</button>
        </div>
      )}
    </div>
  );
}
