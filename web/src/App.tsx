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
import { validatePoseData, ValidationResult, createNpyBuffer } from './lib/validator-utils';
import { Validator } from './Validator';

const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'N/A' },
] as const;

type Gender = (typeof GENDER_OPTIONS)[number]['value'];
type Step = 'home' | 'testing' | 'recording' | 'confirm' | 'validator';
type CameraFacing = 'user' | 'environment';
type CaptureMode = 'pose_only' | 'holistic';
type RecordedPoseQuality = Pick<PoseFrameQuality, 'score' | 'averageVisibility' | 'reliableLandmarks' | 'inFrameLandmarks' | 'bodyBoxArea'> & {
  timestampMs: number;
};

function safeCoord(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
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

async function uploadToSupabaseDirect(payload: any, captureId: string) {
  try {
    const jsonFileName = `${captureId}/raw_capture.json`;
    const npyFileName = `${captureId}/keypoints.npy`;
    
    const jsonBlob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const npyBuffer = createNpyBuffer(payload.keypoints);
    const npyBlob = new Blob([npyBuffer], { type: 'application/octet-stream' });

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

    // 3. Insert DB record
    const { error: dbError } = await supabase.from('captures').insert({
      capture_id: captureId,
      meta: payload.meta,
      storage_paths: {
        raw_json: jsonFileName,
        keypoints_npy: npyFileName
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
      body: JSON.stringify({
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

  // App State
  const [step, setStep] = useState<Step>('home');
  const [isReady, setIsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [skippedFrames, setSkippedFrames] = useState(0);
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
  const [showPermissionGuide, setShowPermissionGuide] = useState(false);
  const cameraActive = step === 'testing' || step === 'recording';

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
    const loop = async () => {
      if (videoRef.current && engineRef.current) {
        await engineRef.current.send(videoRef.current, {
          face: captureMode === 'holistic',
          hands: captureMode === 'holistic'
        });
        requestAnimationFrame(loop);
      }
    };
    loop();
  };

  const handlePoseResults = (results: any) => {
    setLatestResults(results.pose);
    const poseQuality = analyzePoseFrame(results.pose);

    if (isRecordingRef.current) {
      if (!poseQuality.usable || !results.pose?.landmarks?.length) {
        skippedFramesRef.current += 1;
        setSkippedFrames(skippedFramesRef.current);
        return;
      }

      const poseLandmarks = results.pose.landmarks[0];
      const frame = poseLandmarks.map((lm: any) => [
        safeCoord(lm.x), safeCoord(lm.y), safeCoord(lm.z)
      ]);
      const elapsed = Date.now() - startTimeRef.current;
      
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
        faceFramesRef.current.push(results.face.faceLandmarks[0].map((lm: any) => [safeCoord(lm.x), safeCoord(lm.y), safeCoord(lm.z)]));
        faceBlendshapesRef.current.push(results.face.faceBlendshapes?.[0] || []);
      } else {
        // Pad with 478 zeros for consistency
        faceFramesRef.current.push(Array.from({ length: 478 }, () => [0, 0, 0]));
        faceBlendshapesRef.current.push([]);
      }

      if (results.hands && results.hands.landmarks && results.hands.landmarks.length > 0) {
        // We want a fixed 42 points (21 per hand). 
        // If only 1 hand, we pad the rest.
        const landmarks = results.hands.landmarks.flat().map((lm: any) => [safeCoord(lm.x), safeCoord(lm.y), safeCoord(lm.z)]);
        while (landmarks.length < 42) landmarks.push([0, 0, 0]);
        handFramesRef.current.push(landmarks.slice(0, 42));
      } else {
        // Pad with 42 zeros
        handFramesRef.current.push(Array.from({ length: 42 }, () => [0, 0, 0]));
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
    startTimeRef.current = Date.now();
    setFrameCount(0);
    setSkippedFrames(0);
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

    const payload = {
      keypoints: framesRef.current,
      timestamps: timestampsRef.current,
      face_keypoints: captureMode === 'holistic' ? faceFramesRef.current : [],
      face_blendshapes: captureMode === 'holistic' ? faceBlendshapesRef.current : [],
      hand_keypoints: captureMode === 'holistic' ? handFramesRef.current : [],
      quality: {
        pose: summarizePoseQuality(poseQualityRef.current, skippedFramesRef.current),
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
        gender: gender
      }
    };

    let success = false;
    let errorMessage = '';

    if (destination === 'supabase') {
      success = await uploadToSupabaseDirect(payload, captureId);
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
        uploadToSupabaseDirect(payload, captureId),
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
      const data = await res.json();
      if (res.ok) {
        alert('MongoDB Connection Successful!');
      } else {
        alert(`MongoDB Connection Failed!\nReason: ${data.error || data.message}`);
      }
    } catch (err: any) {
      alert(`Network Error: ${err.message}`);
    }
    setIsUploading(false);
  };

  const toggleCamera = () => {
    setCameraFacing(prev => prev === 'user' ? 'environment' : 'user');
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

              <div className="form-grid form-grid-spaced">
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Age</label>
                  <input type="number" className="input-field" value={age} onChange={e => setAge(e.target.value)} placeholder="Age" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Action Type</label>
                  <input className="input-field" value={actionType} onChange={e => setActionType(e.target.value)} placeholder="e.g. sit, stand, reach, turn" />
                </div>
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
                    REC {(duration/1000).toFixed(1)}s | {frameCount} Frames{skippedFrames > 0 ? ` | ${skippedFrames} skipped` : ''}
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
