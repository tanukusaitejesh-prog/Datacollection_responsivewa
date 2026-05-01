import React, { useState, useRef, useEffect } from 'react';
import { 
  FileCheck, 
  Upload, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  Download,
  BarChart3,
  Activity as ActivityIcon,
  ArrowLeft
} from 'lucide-react';
import { 
  validatePoseData, 
  ValidationResult 
} from './lib/validator-utils';

interface ParsedData {
  shape: number[];
  data: Float64Array;
  descr: string;
  fortran: boolean;
}

export const Validator: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [parsed, setParsed] = useState<ParsedData | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleFile = async (file: File) => {
    setFile(file);
    setResult(null);
    setParsed(null);

    const reader = new FileReader();
    if (file.name.endsWith('.json')) {
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string;
          const parsedData = parseJson(text);
          // Convert flat Float64Array back to number[][][] for the utility
          const T = parsedData.shape[0];
          const frames: number[][][] = [];
          for (let t = 0; t < T; t++) {
            const frame: number[][] = [];
            for (let i = 0; i < 33; i++) {
              frame.push([
                parsedData.data[t * 99 + i * 3 + 0],
                parsedData.data[t * 99 + i * 3 + 1],
                parsedData.data[t * 99 + i * 3 + 2]
              ]);
            }
            frames.push(frame);
          }
          const validationResult = validatePoseData(frames);
          setParsed(parsedData);
          setResult(validationResult);
        } catch (err: any) {
          alert('Error parsing JSON: ' + err.message);
        }
      };
      reader.readAsText(file);
    } else if (file.name.endsWith('.npy')) {
      reader.onload = (e) => {
        try {
          const buffer = e.target?.result as ArrayBuffer;
          const parsedData = parseNpy(buffer);
          // Convert flat Float64Array back to number[][][] for the utility
          const T = parsedData.shape[0];
          const frames: number[][][] = [];
          for (let t = 0; t < T; t++) {
            const frame: number[][] = [];
            for (let i = 0; i < 33; i++) {
              frame.push([
                parsedData.data[t * 99 + i * 3 + 0],
                parsedData.data[t * 99 + i * 3 + 1],
                parsedData.data[t * 99 + i * 3 + 2]
              ]);
            }
            frames.push(frame);
          }
          const validationResult = validatePoseData(frames);
          setParsed(parsedData);
          setResult(validationResult);
        } catch (err: any) {
          alert('Error parsing NPY: ' + err.message);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      alert('Please upload a .npy or .json file');
    }
  };

  useEffect(() => {
    if (result && parsed && canvasRef.current) {
      drawSkeleton(canvasRef.current, parsed.data, parsed.shape);
    }
  }, [result, parsed]);

  const parseJson = (text: string): ParsedData => {
    const obj = JSON.parse(text);
    let frames: number[][] = [];
    let arr = Array.isArray(obj) ? obj : null;
    const toCoord = (value: any) => {
      if (value === 'NaN') return Number.NaN;
      return typeof value === 'number' && Number.isFinite(value) ? value : Number.NaN;
    };
    if (!arr && typeof obj === 'object') {
      if (Array.isArray(obj.data)) arr = obj.data;
      else if (Array.isArray(obj.frames)) arr = obj.frames;
      else if (Array.isArray(obj.pose_data)) arr = obj.pose_data;
      else if (Array.isArray(obj.landmarks)) arr = obj.landmarks;
      else if (Array.isArray(obj.keypoints)) arr = obj.keypoints;
    }
    if (!arr || arr.length === 0) throw new Error("Could not find an array of frames in the JSON");
    
    for (let row of arr) {
      let landmarks: any = null;
      if (typeof row === 'string') { try { row = JSON.parse(row); } catch(e){} }
      if (Array.isArray(row) && row.length === 33) landmarks = row;
      else if (Array.isArray(row) && row.length > 0 && Array.isArray(row[0]) && row[0].length >= 3) landmarks = row;
      else if (row.pose_data) {
        let pd = typeof row.pose_data === 'string' ? JSON.parse(row.pose_data) : row.pose_data;
        if (Array.isArray(pd)) landmarks = pd;
      } else if (row.landmarks) landmarks = row.landmarks;
      else if (row.pose) landmarks = row.pose;
      
      if (landmarks && landmarks.length === 33) {
        let flat = [];
        for (let i = 0; i < 33; i++) {
          let lm = landmarks[i];
          if (Array.isArray(lm)) { flat.push(toCoord(lm[0]), toCoord(lm[1]), toCoord(lm[2])); }
          else if (typeof lm === 'object') { flat.push(toCoord(lm.x), toCoord(lm.y), toCoord(lm.z)); }
          else { flat.push(Number.NaN, Number.NaN, Number.NaN); }
        }
        frames.push(flat);
      }
    }
    
    if (frames.length === 0) throw new Error("Could not extract any 33-landmark frames from the JSON");
    const T = frames.length;
    const data = new Float64Array(T * 33 * 3);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < 99; i++) {
        data[t * 99 + i] = frames[t][i];
      }
    }
    return { shape: [T, 33, 3], data, descr: '<f8', fortran: false };
  };

  const parseNpy = (buffer: ArrayBuffer): ParsedData => {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(dv.getUint8(0),dv.getUint8(1),dv.getUint8(2),dv.getUint8(3),dv.getUint8(4),dv.getUint8(5));
    if(!magic.includes('NUMPY')) throw new Error('Not a valid .npy file');
    const major = dv.getUint8(6);
    let headerLen, dataOffset;
    if(major >= 2){ headerLen = dv.getUint32(8, true); dataOffset = 12 + headerLen; }
    else{ headerLen = dv.getUint16(8, true); dataOffset = 10 + headerLen; }
    const headerStr = new TextDecoder().decode(new Uint8Array(buffer, major >= 2 ? 12 : 10, headerLen));
    
    const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
    if(!shapeMatch) throw new Error('Cannot parse shape from header');
    const shape = shapeMatch[1].split(',').map(s=>s.trim()).filter(s=>s.length>0).map(Number);
    
    const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']*)'/);
    const descr = descrMatch ? descrMatch[1] : '<f8';
    const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
    const fortran = fortranMatch ? fortranMatch[1] === 'True' : false;
    
    let bytesPerElem, reader: (o: number) => number;
    const dt = descr.replace(/[<>=|]/,'');
    if(dt === 'f4'){ bytesPerElem = 4; reader = (o) => dv.getFloat32(o, true); }
    else if(dt === 'f8'){ bytesPerElem = 8; reader = (o) => dv.getFloat64(o, true); }
    else if(dt.startsWith('i4')){ bytesPerElem = 4; reader = (o) => dv.getInt32(o, true); }
    else { bytesPerElem = 8; reader = (o) => dv.getFloat64(o, true); }
    
    const totalElems = shape.reduce((a,b) => a*b, 1);
    const data = new Float64Array(totalElems);
    for(let i=0; i<totalElems; i++) data[i] = reader(dataOffset + i*bytesPerElem);
    
    return { shape, data, descr, fortran };
  };


  const drawSkeleton = (canvas: HTMLCanvasElement, data: Float64Array, shape: number[]) => {
    if(shape.length !== 3 || shape[1] !== 33 || shape[2] !== 3) return;
    const ctx = canvas.getContext('2d');
    if(!ctx) return;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    
    const pts: {x: number, y: number}[] = [];
    for(let j=0; j<33; j++){
      const x = data[j*3], y = data[j*3+1];
      pts.push({x, y});
    }

    let mnx=Infinity, mxx=-Infinity, mny=Infinity, mxy=-Infinity;
    pts.forEach(p=>{
      if(!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
      if(p.x<mnx) mnx=p.x; if(p.x>mxx) mxx=p.x; if(p.y<mny) mny=p.y; if(p.y>mxy) mxy=p.y;
    });
    if (!Number.isFinite(mnx) || !Number.isFinite(mxx) || !Number.isFinite(mny) || !Number.isFinite(mxy)) return;
    const pad = 40; 
    const scaleX = (W - pad*2) / (mxx-mnx || 1); 
    const scaleY = (H - pad*2) / (mxy-mny || 1);
    const scale = Math.min(scaleX, scaleY);
    const cx = W/2, cy = H/2;
    const ox = (mnx+mxx)/2, oy = (mny+mxy)/2;
    
    const tx = (p: {x: number, y: number}) => cx + (p.x-ox)*scale;
    const ty = (p: {x: number, y: number}) => cy + (p.y-oy)*scale;

    const conns = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[12,14],[14,16],[15,17],[15,19],[15,21],[16,18],[16,20],[16,22],[17,19],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
    ctx.strokeStyle = 'rgba(14, 106, 168, 0.5)';
    ctx.lineWidth = 2;
    conns.forEach(([a,b]) => {
      if (!Number.isFinite(pts[a].x) || !Number.isFinite(pts[a].y) || !Number.isFinite(pts[b].x) || !Number.isFinite(pts[b].y)) return;
      ctx.beginPath();
      ctx.moveTo(tx(pts[a]), ty(pts[a]));
      ctx.lineTo(tx(pts[b]), ty(pts[b]));
      ctx.stroke();
    });

    pts.forEach((p, i) => {
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
      ctx.beginPath();
      ctx.arc(tx(p), ty(p), 4, 0, Math.PI*2);
      ctx.fillStyle = i < 11 ? '#B74E63' : (i < 23 ? '#0E6AA8' : '#1F8A6D');
      ctx.fill();
    });
  };

  const handleExportNpy = () => {
    if(!parsed || !file) return;
    const { shape, data } = parsed;
    const magic = new Uint8Array([147, 78, 85, 77, 80, 89]);
    const dictStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + shape.join(', ') + "), }";
    let totalLen = 10 + dictStr.length + 1;
    let paddingLen = 64 - (totalLen % 64);
    if (paddingLen === 64) paddingLen = 0;
    const headerStr = dictStr + ' '.repeat(paddingLen) + '\n';
    const headerLen = headerStr.length;
    const buffer = new ArrayBuffer(10 + headerLen + data.byteLength);
    const dv = new DataView(buffer);
    for(let i=0; i<6; i++) dv.setUint8(i, magic[i]);
    dv.setUint8(6, 1);
    dv.setUint8(7, 0);
    dv.setUint16(8, headerLen, true);
    for(let i=0; i<headerLen; i++) dv.setUint8(10 + i, headerStr.charCodeAt(i));
    
    const outBytes = new Uint8Array(buffer);
    const dataBytes = new Uint8Array(data.buffer);
    outBytes.set(dataBytes, 10 + headerLen);
    
    const url = URL.createObjectURL(new Blob([buffer], { type: 'application/octet-stream' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = file.name.replace('.json', '') + '.npy';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      <div className="step-container">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
          <button onClick={onBack} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 4, display: 'flex', alignItems: 'center', color: 'var(--text-secondary)' }}>
            <ArrowLeft size={20} />
          </button>
          <h1 className="home-title" style={{ margin: 0 }}>NPY Validator</h1>
        </div>

        <div 
          className={`card ${isDragging ? 'dragover' : ''}`}
          style={{ 
            border: '2px dashed var(--border)', 
            textAlign: 'center', 
            padding: '40px 20px',
            cursor: 'pointer',
            backgroundColor: isDragging ? 'var(--accent-light)' : 'transparent',
            transition: 'all 0.2s'
          }}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setIsDragging(false);
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
          }}
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <Upload size={40} style={{ color: 'var(--accent)', marginBottom: 12 }} />
          <p style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {file ? file.name : 'Upload .npy or .json file'}
          </p>
          <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
            Drag & drop or click to browse
          </p>
          <input 
            id="file-upload" 
            type="file" 
            accept=".npy,.json" 
            style={{ display: 'none' }} 
            onChange={(e) => {
              if (e.target.files?.length) handleFile(e.target.files[0]);
            }}
          />
        </div>

        {result && (
          <div style={{ marginTop: 20, animation: 'fadeUp 0.4s ease' }}>
            <div className={`card`} style={{ 
              backgroundColor: result.overall === 'pass' ? '#ecfdf5' : result.overall === 'warn' ? '#fffbeb' : '#fef2f2',
              borderColor: result.overall === 'pass' ? '#10b981' : result.overall === 'warn' ? '#f59e0b' : '#ef4444',
              textAlign: 'center',
              padding: '24px'
            }}>
              {result.overall === 'pass' && <CheckCircle2 size={32} color="#10b981" style={{ margin: '0 auto 8px' }} />}
              {result.overall === 'warn' && <AlertTriangle size={32} color="#f59e0b" style={{ margin: '0 auto 8px' }} />}
              {result.overall === 'fail' && <XCircle size={32} color="#ef4444" style={{ margin: '0 auto 8px' }} />}
              
              <h2 style={{ fontSize: 18, color: result.overall === 'pass' ? '#065f46' : result.overall === 'warn' ? '#92400e' : '#991b1b' }}>
                {result.overall === 'pass' ? 'Ready for Training' : result.overall === 'warn' ? 'Usable with Warnings' : 'Not Fit for Training'}
              </h2>
              <p style={{ fontSize: 13, color: 'rgba(0,0,0,0.6)', marginTop: 4 }}>
                {result.overall === 'pass' ? 'This file meets all requirements.' : result.overall === 'warn' ? `Has ${result.warns} warning(s).` : `Failed ${result.fails} critical check(s).`}
              </p>

              {file?.name.endsWith('.json') && result.overall !== 'fail' && (
                <button 
                  className="btn btn-primary" 
                  style={{ marginTop: 16, maxWidth: 240, margin: '16px auto 0' }}
                  onClick={handleExportNpy}
                >
                  <Download size={16} /> Export as .npy
                </button>
              )}
            </div>

            <div className="card">
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <BarChart3 size={16} /> File Statistics
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
                {[
                  ['Shape', result.stats.shape],
                  ['Frames', result.stats.frames],
                  ['Landmarks', result.stats.landmarks],
                  ['Clips', result.stats.clips],
                  ['Mean', result.stats.mean],
                  ['Std Dev', result.stats.std]
                ].map(([label, val]) => (
                  <div key={label} style={{ padding: 10, backgroundColor: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <FileCheck size={16} /> Validation Checks
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
                {result.checks.map((c, i) => (
                  <div key={i} style={{ display: 'flex', gap: 12, padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <div style={{ 
                      width: 20, height: 20, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                      backgroundColor: c.s === 'pass' ? '#d1fae5' : c.s === 'warn' ? '#fef3c7' : '#fee2e2',
                      color: c.s === 'pass' ? '#059669' : c.s === 'warn' ? '#d97706' : '#dc2626',
                      fontSize: 10, fontWeight: 900, flexShrink: 0
                    }}>
                      {c.s === 'pass' ? '✓' : c.s === 'warn' ? '!' : '✗'}
                    </div>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600 }}>{c.name}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{c.detail}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card" style={{ textAlign: 'center' }}>
              <div className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                <ActivityIcon size={16} /> Skeleton Preview (Frame 0)
              </div>
              <canvas 
                ref={canvasRef} 
                width={300} 
                height={300} 
                style={{ backgroundColor: '#f8fafc', borderRadius: 12, marginTop: 12, maxWidth: '100%' }}
              />
              <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 8 }}>
                33 Pose landmarks detected
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
