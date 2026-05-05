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
import { validatePoseData, ValidationResult, createNpyBuffer, normalizePoseSequence } from './lib/validator-utils';
import { Validator } from './Validator';

const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'N/A' },
] as const;

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
type CaptureMode = 'pose_only' | 'holistic';
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

  const jointNames = [
    'Midspain', 'AnkleLeft', 'AnkleRight', 'ElbowLeft', 'ElbowRight',
    'FootLeft', 'FootRight', 'HandLeft', 'HandRight', 'HandTipLeft',
    'HandTipRight', 'Head', 'HipLeft', 'HipRight', 'KneeLeft',
    'KneeRight', 'Neck', 'ShoulderLeft', 'ShoulderRight', 'SpineBase',
    'SpineShoulder', 'ThumbLeft', 'ThumbRight', 'WristLeft', 'WristRight'
  ];

  const featureNames = [
    'HESHL', 'HESHR', 'SPELL', 'SPELR', 'SHWRL', 'SHWRR', 'ELHAL', 'ELHAR',
    'THHAL', 'THHAR', 'THHTIL', 'THHTIR', 'SPKNL', 'SPKNR', 'HIANL', 'HIANR',
    'KNFOL', 'KNFOR', 'DFRToFL', 'MinDBFAC', 'MaxDBFE', 'MinDBFE', 'Threshold'
  ];

  const headers = ['H:M:S:MS'];
  jointNames.forEach(name => headers.push(`${name}_X`, `${name}_Y`, `${name}_Z`));
  featureNames.forEach(name => headers.push(name));

  const csvRows = [headers.join('\t')];

  const getAngle = (p1: number[], p2: number[], p3: number[]) => {
    if (!p1 || !p2 || !p3) return 0;
    const v1 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    const v2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
    const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    const mag1 = Math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2);
    const mag2 = Math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2);
    if (mag1 * mag2 < 1e-6) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2)))) * (180 / Math.PI);
  };

  const getDist = (p1: number[], p2: number[]) => {
    if (!p1 || !p2) return 0;
    return Math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2);
  };

  const startTime = Date.now(); // Base time for H:M:S:MS

  frames.forEach((frame, t) => {
    const ts = timestamps[t] || 0;
    const d = new Date(startTime + ts);
    const timeStr = `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}:${d.getMilliseconds().toString().padStart(3, '0')}`;

    // Map 33 MP landmarks to 25 Kinect joints
    const joints: Record<string, number[]> = {};
    const MP = (i: number) => frame[i] || [0,0,0];

    const midHips = [(MP(23)[0] + MP(24)[0])/2, (MP(23)[1] + MP(24)[1])/2, (MP(23)[2] + MP(24)[2])/2];
    const midShoulders = [(MP(11)[0] + MP(12)[0])/2, (MP(11)[1] + MP(12)[1])/2, (MP(11)[2] + MP(12)[2])/2];

    joints['Midspain'] = [(midHips[0] + midShoulders[0])/2, (midHips[1] + midShoulders[1])/2, (midHips[2] + midShoulders[2])/2];
    joints['AnkleLeft'] = MP(27);
    joints['AnkleRight'] = MP(28);
    joints['ElbowLeft'] = MP(13);
    joints['ElbowRight'] = MP(14);
    joints['FootLeft'] = MP(31);
    joints['FootRight'] = MP(32);
    joints['HandLeft'] = MP(15);
    joints['HandRight'] = MP(16);
    joints['HandTipLeft'] = MP(19);
    joints['HandTipRight'] = MP(20);
    joints['Head'] = MP(0);
    joints['HipLeft'] = MP(23);
    joints['HipRight'] = MP(24);
    joints['KneeLeft'] = MP(25);
    joints['KneeRight'] = MP(26);
    joints['Neck'] = [(midShoulders[0]*0.8 + MP(0)[0]*0.2), (midShoulders[1]*0.8 + MP(0)[1]*0.2), (midShoulders[2]*0.8 + MP(0)[2]*0.2)];
    joints['ShoulderLeft'] = MP(11);
    joints['ShoulderRight'] = MP(12);
    joints['SpineBase'] = midHips;
    joints['SpineShoulder'] = midShoulders;
    joints['ThumbLeft'] = MP(21);
    joints['ThumbRight'] = MP(22);
    joints['WristLeft'] = MP(15);
    joints['WristRight'] = MP(16);

    const row: any[] = [timeStr];
    jointNames.forEach(name => {
      const p = joints[name];
      row.push(p[0].toFixed(6), p[1].toFixed(6), p[2].toFixed(6));
    });

    // Derived Features
    const f: Record<string, number> = {};
    f['HESHL'] = getAngle(joints['Head'], joints['Neck'], joints['ShoulderLeft']);
    f['HESHR'] = getAngle(joints['Head'], joints['Neck'], joints['ShoulderRight']);
    f['SPELL'] = getAngle(joints['SpineShoulder'], joints['ShoulderLeft'], joints['ElbowLeft']);
    f['SPELR'] = getAngle(joints['SpineShoulder'], joints['ShoulderRight'], joints['ElbowRight']);
    f['SHWRL'] = getAngle(joints['ShoulderLeft'], joints['ElbowLeft'], joints['WristLeft']);
    f['SHWRR'] = getAngle(joints['ShoulderRight'], joints['ElbowRight'], joints['WristRight']);
    f['ELHAL'] = getAngle(joints['ElbowLeft'], joints['ShoulderLeft'], joints['HipLeft']);
    f['ELHAR'] = getAngle(joints['ElbowRight'], joints['ShoulderRight'], joints['HipRight']);
    f['THHAL'] = getAngle(joints['ThumbLeft'], joints['WristLeft'], joints['HandLeft']);
    f['THHAR'] = getAngle(joints['ThumbRight'], joints['WristRight'], joints['HandRight']);
    f['THHTIL'] = getAngle(joints['ThumbLeft'], joints['HandLeft'], joints['HandTipLeft']);
    f['THHTIR'] = getAngle(joints['ThumbRight'], joints['HandRight'], joints['HandTipRight']);
    f['SPKNL'] = getAngle(joints['SpineBase'], joints['HipLeft'], joints['KneeLeft']);
    f['SPKNR'] = getAngle(joints['SpineBase'], joints['HipRight'], joints['KneeRight']);
    f['HIANL'] = getAngle(joints['HipLeft'], joints['KneeLeft'], joints['AnkleLeft']);
    f['HIANR'] = getAngle(joints['HipRight'], joints['KneeRight'], joints['AnkleRight']);
    f['KNFOL'] = getAngle(joints['KneeLeft'], joints['AnkleLeft'], joints['FootLeft']);
    f['KNFOR'] = getAngle(joints['KneeRight'], joints['AnkleRight'], joints['FootRight']);
    f['DFRToFL'] = getDist(joints['SpineBase'], joints['FootLeft']);
    f['MinDBFAC'] = getDist(joints['AnkleLeft'], joints['AnkleRight']);
    f['MaxDBFE'] = f['MinDBFAC']; // Placeholder for max/min logic
    f['MinDBFE'] = f['MinDBFAC'];
    f['Threshold'] = 0.3;

    featureNames.forEach(name => row.push(f[name].toFixed(6)));
    csvRows.push(row.join('\t'));
  });

  return csvRows.join('\n');
}

async function uploadToSupabaseDirect(payload: any, captureId: string, csvString: string) {
  try {
    const jsonFileName = `${captureId}/raw_capture.json`;
    const npyFileName = `${captureId}/keypoints.npy`;
    const csvFileName = `${captureId}/keypoints.csv`;
    
    const jsonBlob = new Blob([stringifyWithNaN(payload)], { type: 'application/json' });
    const npyBuffer = createNpyBuffer(payload.keypoints);
    const npyBlob = new Blob([npyBuffer], { type: 'application/octet-stream' });
    const csvBlob = new Blob([csvString], { type: 'text/csv' });

    // 1. Upload JSON
    const { error: jsonError } = await supabase.storage
      .from('pose-captures')
      .upload(jsonFileName, jsonBlob, { upsert: true });

    if (jsonError) throw jsonError;

    // 2. Upload NPY
    const { error: npyError } = await supabase.storage
      .from('pose-captures')
      .upload(npyFileName, npyBlob, { upsert: true });

    if (npyError) throw npyError;

    // 3. Upload CSV
    const { error: csvError } = await supabase.storage
      .from('pose-captures')
      .upload(csvFileName, csvBlob, { upsert: true });

    if (csvError) throw csvError;

    // 4. Insert DB record
    const { error: dbError } = await supabase.from('captures').insert({
      capture_id: captureId,
      meta: payload.meta,
      storage_paths: {
        raw_json: jsonFileName,
        keypoints_npy: npyFileName,
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

async function pushToMongoDirect(payload: any, captureId: string) {
  try {
    const response = await fetch('/.netlify/functions/pushToMongo', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: stringifyWithNaN({
        captureId,
        ...payload
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || errorData.message || 'Failed to push to MongoDB');
    }

    return true;
  } catch (err: any) {
    console.error('Web MongoDB direct push failed:', err);
    let message = err.message;
    if (message.includes('Unexpected token') || message.includes('DOCTYPE')) {
      message = 'Netlify function not found. Use "netlify dev" to run locally.';
    }
    return { success: false, error: message };
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

  // App State
  const [step, setStep] = useState<Step>('home');
  const [isReady, setIsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [skippedFrames, setSkippedFrames] = useState(0);
  const [faceDetectedFrames, setFaceDetectedFrames] = useState(0);
  const [handDetectedFrames, setHandDetectedFrames] = useState(0);
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

      if (results.hands && results.hands.landmarks && results.hands.landmarks.length > 0) {
        const landmarks = results.hands.landmarks.flat().map((lm: any) => [coordOrNaN(lm.x), coordOrNaN(lm.y), coordOrNaN(lm.z)]);
        while (landmarks.length < HAND_LANDMARK_COUNT) landmarks.push(missingPoint());
        handFramesRef.current.push(landmarks.slice(0, HAND_LANDMARK_COUNT));
        handDetectedFramesRef.current += 1;
        setHandDetectedFrames(handDetectedFramesRef.current);
      } else {
        handFramesRef.current.push(missingLandmarkFrame(HAND_LANDMARK_COUNT));
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
    isRecordingRef.current = true;
    setIsRecording(true);
    setStep('recording');
  };

  const stopRecording = () => {
    isRecordingRef.current = false;
    setIsRecording(false);
    const result = validatePoseData(framesRef.current);
    setValidationResult(result);
    setStep('confirm');
  };

  const handleFinalize = async (destination: 'supabase' | 'mongo' | 'both') => {
    setIsUploading(true);
    
    const captureId = `web_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    const safeAge = parseInt(age);
    const lastTs = timestampsRef.current[timestampsRef.current.length - 1] || 1;
    const actualFps = framesRef.current.length / (lastTs / 1000);
    const recordedFrames = framesRef.current.length;
    const normalizedFrames = normalizePoseSequence(framesRef.current);
    const csvString = generateClinicalCsvString(normalizedFrames, timestampsRef.current);

    const payload = {
      keypoints: normalizedFrames,
      csv_data: csvString,
      timestamps: timestampsRef.current,
      face_keypoints: captureMode === 'holistic' ? faceFramesRef.current : missingLandmarkFrames(recordedFrames, FACE_LANDMARK_COUNT),
      face_blendshapes: captureMode === 'holistic' ? faceBlendshapesRef.current : missingBlendshapeFrames(recordedFrames),
      hand_keypoints: captureMode === 'holistic' ? handFramesRef.current : missingLandmarkFrames(recordedFrames, HAND_LANDMARK_COUNT),
      quality: {
        pose: summarizePoseQuality(poseQualityRef.current, skippedFramesRef.current),
        holistic: {
          face_detected_frames: faceDetectedFramesRef.current,
          hand_detected_frames: handDetectedFramesRef.current
        },
        validation: validationResult
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
        clinician_notes: clinicianNotes
      }
    };

    let success = false;
    let errorMessage = '';

    if (destination === 'supabase') {
      success = await uploadToSupabaseDirect(payload, captureId, csvString);
      if (!success) errorMessage = 'Unknown Supabase error';
    } else if (destination === 'mongo') {
      const result = await pushToMongoDirect(payload, captureId);
      if (result === true) {
        success = true;
      } else {
        success = false;
        errorMessage = (result as any).error || 'Unknown MongoDB error';
      }
    } else if (destination === 'both') {
      const [sRes, mRes] = await Promise.all([
        uploadToSupabaseDirect(payload, captureId, csvString),
        pushToMongoDirect(payload, captureId)
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
    const normalizedFrames = normalizePoseSequence(framesRef.current);
    const csvString = generateClinicalCsvString(normalizedFrames, timestampsRef.current);
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
                <div style={{display: 'flex', gap: 8, marginTop: 4}}>
                  <button 
                    className={`chip ${captureMode === 'pose_only' ? 'active' : ''}`}
                    style={{flex: 1, textAlign: 'center'}}
                    onClick={() => setCaptureMode('pose_only')}
                  >
                    Pose Only
                  </button>
                  <button 
                    className={`chip ${captureMode === 'holistic' ? 'active' : ''}`}
                    style={{flex: 1, textAlign: 'center'}}
                    onClick={() => setCaptureMode('holistic')}
                  >
                    Holistic
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
