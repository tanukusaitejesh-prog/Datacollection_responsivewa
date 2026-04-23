import { useEffect, useRef, useState } from 'react';
import { PoseEngine } from './lib/pose-engine';
import { 
  Square, 
  AlertCircle, 
  RefreshCw,
  ArrowLeft,
  ChevronRight,
  UploadCloud,
  FlipHorizontal
} from 'lucide-react';
import { PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import { supabase } from './lib/supabase';
import { evaluateCaptureReadiness } from './lib/quality';

const GENDER_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'other', label: 'Other' },
  { value: 'prefer_not_to_say', label: 'N/A' },
] as const;

type Gender = (typeof GENDER_OPTIONS)[number]['value'];
type Step = 'home' | 'testing' | 'recording' | 'confirm';
type CameraFacing = 'user' | 'environment';

async function uploadToSupabaseDirect(payload: any, captureId: string) {
  try {
    const fileName = `${captureId}/raw_capture.json`;
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });

    const { error: storageError } = await supabase.storage
      .from('pose-captures')
      .upload(fileName, blob, {
        upsert: true
      });

    if (storageError) throw storageError;

    const { error: dbError } = await supabase.from('captures').insert({
      capture_id: captureId,
      meta: payload.meta,
      storage_paths: {
        raw_json: fileName
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
      throw new Error(errorData.message || 'Failed to push to MongoDB');
    }

    return true;
  } catch (err) {
    console.error('Web MongoDB direct push failed:', err);
    return false;
  }
}

export default function App() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<PoseEngine | null>(null);
  
  // Recording Data
  const framesRef = useRef<number[][][]>([]);
  const timestampsRef = useRef<number[]>([]);
  const startTimeRef = useRef<number>(0);

  // App State
  const [step, setStep] = useState<Step>('home');
  const [isReady, setIsReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [duration, setDuration] = useState(0);
  const [latestResults, setLatestResults] = useState<PoseLandmarkerResult | null>(null);

  // Metadata State
  const [sessionId, setSessionId] = useState('');
  const [subjectName, setSubjectName] = useState('');
  const [subjectId, setSubjectId] = useState('');
  const [actionType, setActionType] = useState('');
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<Gender>('prefer_not_to_say');
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>('user');
  const [showPermissionGuide, setShowPermissionGuide] = useState(false);

  useEffect(() => {
    if (step === 'testing' || step === 'recording') {
      initCamera();
    }
    return () => {
      stopCamera();
    };
  }, [step, cameraFacing]);

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    engineRef.current?.close();
    engineRef.current = null;
    setIsReady(false);
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
        await engineRef.current.send(videoRef.current);
        requestAnimationFrame(loop);
      }
    };
    loop();
  };

  const handlePoseResults = (results: PoseLandmarkerResult) => {
    setLatestResults(results);
    
    if (isRecording && results.landmarks && results.landmarks.length > 0) {
      const poseLandmarks = results.landmarks[0];
      const frame = poseLandmarks.map(lm => [
        lm.x, lm.y, lm.z, lm.visibility ?? 0
      ]);
      
      framesRef.current.push(frame);
      const elapsed = Date.now() - startTimeRef.current;
      timestampsRef.current.push(elapsed);
      setFrameCount(framesRef.current.length);
      setDuration(elapsed);
    }
  };

  const startRecording = () => {
    framesRef.current = [];
    timestampsRef.current = [];
    startTimeRef.current = Date.now();
    setFrameCount(0);
    setDuration(0);
    setIsRecording(true);
    setStep('recording');
  };

  const stopRecording = () => {
    setIsRecording(false);
    setStep('confirm');
  };

  const handleFinalize = async (destination: 'supabase' | 'mongo') => {
    setIsUploading(true);
    
    const captureId = `web_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
    const safeAge = parseInt(age);
    const lastTs = timestampsRef.current[timestampsRef.current.length - 1] || 1;
    const actualFps = framesRef.current.length / (lastTs / 1000);

    const payload = {
      keypoints: framesRef.current,
      timestamps: timestampsRef.current,
      meta: {
        fps_nominal: isFinite(actualFps) ? actualFps : 30,
        resolution: [1280, 720],
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
    if (destination === 'supabase') {
      success = await uploadToSupabaseDirect(payload, captureId);
    } else if (destination === 'mongo') {
      success = await pushToMongoDirect(payload, captureId);
    }
    
    setIsUploading(false);

    if (success) {
      alert(`Upload to ${destination === 'supabase' ? 'Cloud' : 'MongoDB'} Successful!`);
      setStep('home');
    } else {
      alert(`Upload to ${destination === 'supabase' ? 'Cloud' : 'MongoDB'} Failed. Check your connection.`);
    }
  };

  const toggleCamera = () => {
    setCameraFacing(prev => prev === 'user' ? 'environment' : 'user');
  };

  const readiness = evaluateCaptureReadiness(latestResults, isReady);

  return (
    <div className="app-root">
      {step === 'home' && (
        <div className="app-container">
          <div className="step-container">
            <h1 className="home-title">Pose Capture Studio</h1>
            <p className="home-subtitle">Configure subject details to begin.</p>

            <div className="card">
              <div className="card-title">Subject Profile</div>
              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10}}>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Full Name</label>
                  <input className="input-field" value={subjectName} onChange={e => setSubjectName(e.target.value)} placeholder="Name" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Subject ID</label>
                  <input className="input-field" value={subjectId} onChange={e => setSubjectId(e.target.value)} placeholder="ID" />
                </div>
              </div>

              <div style={{marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10}}>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Age</label>
                  <input type="number" className="input-field" value={age} onChange={e => setAge(e.target.value)} placeholder="Age" />
                </div>
                <div>
                  <label className="input-label" style={{fontSize: 11, fontWeight: 700, color: '#444'}}>Action Type</label>
                  <input className="input-field" value={actionType} onChange={e => setActionType(e.target.value)} placeholder="e.g. Walking" />
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
            </div>

            <button 
              className="btn btn-primary" 
              disabled={!age || !subjectName}
              onClick={() => setStep('testing')}
            >
              Next: Camera Check <ChevronRight size={18} />
            </button>
          </div>
        </div>
      )}

      {step === 'testing' && (
        <div className="app-container">
          <div className="step-container">
            <div className="camera-wrapper">
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
            </div>

            <div className="testing-panel">
              <div className="card-title">Pre-Capture Check</div>
              {readiness.checks.map(c => (
                <p key={c.id} className={`check-item ${c.ok ? 'check-ok' : 'check-fail'}`}>
                  {c.ok ? '??' : '??'} {c.label}: {c.detail}
                </p>
              ))}
              <p style={{marginTop: 8, fontWeight: 700, fontSize: 13, color: readiness.ready ? 'var(--success)' : 'var(--error)'}}>
                {readiness.summary}
              </p>
            </div>

            <div style={{display: 'flex', gap: 10, marginTop: 'auto', paddingBottom: 20}}>
              <button className="btn btn-secondary" onClick={() => setStep('home')}>
                <ArrowLeft size={18} /> Back
              </button>
              <button className="btn btn-primary" disabled={!readiness.ready} onClick={startRecording}>
                Start Recording
              </button>
            </div>
          </div>
        </div>
      )}

      {step === 'recording' && (
        <div className="app-container" style={{maxWidth: 'none', padding: 0}}>
          <div className="step-container" style={{padding: 0, height: '100vh', position: 'relative'}}>
            <div className="camera-wrapper" style={{height: '100%', borderRadius: 0, margin: 0}}>
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
              
              <div className="recording-bar">
                REC {(duration/1000).toFixed(1)}s | {frameCount} Frames
              </div>

              <div className="stop-btn-overlay">
                <button className="btn btn-danger" onClick={stopRecording}>
                  <Square size={20} fill="currentColor" /> Stop Recording
                </button>
              </div>
            </div>
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
            </div>

            <div className="card">
              <div className="card-title">Edit Metadata</div>
              
              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10}}>
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Name</label>
                  <input className="input-field" value={subjectName} onChange={e => setSubjectName(e.target.value)} />
                </div>
                <div>
                  <label style={{fontSize: 10, fontWeight: 700}}>Subject ID</label>
                  <input className="input-field" value={subjectId} onChange={e => setSubjectId(e.target.value)} />
                </div>
              </div>

              <div style={{marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10}}>
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
                  disabled={isUploading}
                  onClick={() => handleFinalize('supabase')}
                >
                  {isUploading ? <RefreshCw className="animate-spin" /> : <UploadCloud />}
                  {isUploading ? 'Finalizing...' : 'Finalize & Send to Cloud'}
                </button>
                <button 
                  className="btn btn-primary" 
                  style={{ backgroundColor: '#47A248' }} // MongoDB green color
                  disabled={isUploading}
                  onClick={() => handleFinalize('mongo')}
                >
                  {isUploading ? <RefreshCw className="animate-spin" /> : <UploadCloud />}
                  {isUploading ? 'Finalizing...' : 'Finalize & Send to MongoDB'}
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
