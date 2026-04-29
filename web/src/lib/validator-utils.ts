export interface ValidationResult {
  overall: 'pass' | 'warn' | 'fail';
  checks: {
    s: 'pass' | 'warn' | 'fail';
    name: string;
    detail: string;
  }[];
  stats: {
    shape: string;
    frames: number;
    landmarks: number;
    clips: number;
    mean: string;
    std: string;
    range: string;
  };
  fails: number;
  warns: number;
}

export function validatePoseData(frames: number[][][]): ValidationResult {
  const T = frames.length;
  const numLandmarks = frames[0]?.length || 0;
  
  // Flatten frames to [T * 33 * 3] for easier math (similar to NPY format)
  // Our web frames are [T, 33, 4] where 4 is [x, y, z, visibility]
  const data = new Float64Array(T * 33 * 3);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < 33; i++) {
      const lm = frames[t][i] || [0, 0, 0, 0];
      data[t * 99 + i * 3 + 0] = lm[0];
      data[t * 99 + i * 3 + 1] = lm[1];
      data[t * 99 + i * 3 + 2] = lm[2];
    }
  }

  const checks: any[] = [];
  let fails = 0, warns = 0;

  const pass = (name: string, detail: string) => checks.push({s:'pass', name, detail});
  const fail = (name: string, detail: string) => { checks.push({s:'fail', name, detail}); fails++; };
  const warn = (name: string, detail: string) => { checks.push({s:'warn', name, detail}); warns++; };

  // 1. Shape
  if (numLandmarks === 33) {
    pass('Landmarks count is 33', `Detected exactly 33 MediaPipe landmarks.`);
  } else {
    fail('Invalid landmarks count', `Expected 33, got ${numLandmarks}.`);
  }

  // 2. Minimum frames
  if(T >= 48) pass(`Frames ≥ 48 (WINDOW_SIZE)`, `${T} frames → ${Math.floor((T-48)/16)+1} full clips`);
  else if(T >= 32) warn(`Frames ≥ 32 but < 48`, `${T} frames — will be padded.`);
  else fail(`Too few frames (< 32)`, `Only ${T} frames. Minimum is 32.`);

  // 3. NaN check
  let nanCount = 0; for(let i=0; i<data.length; i++) if(isNaN(data[i])) nanCount++;
  if(nanCount === 0) pass('No NaN values', 'All values are valid numbers');
  else if(nanCount/data.length < 0.05) warn(`${nanCount} NaN values`, `Small amount detected.`);
  else fail(`${nanCount} NaN values`, `Too many missing landmarks.`);

  // 4. Zero count (Visibility check)
  let zeroCount = 0; for(let i=0; i<data.length; i++) if(data[i] === 0) zeroCount++;
  const zeroPct = zeroCount/data.length;
  if(zeroPct < 0.15) pass('Good visibility', `${(zeroPct*100).toFixed(1)}% zeros`);
  else if(zeroPct < 0.5) warn('Partial occlusion', `${(zeroPct*100).toFixed(1)}% zeros — some body parts missing.`);
  else fail('Poor visibility', `${(zeroPct*100).toFixed(1)}% zeros — skeleton barely detected.`);

  // 5. Frozen skeleton (temporal std)
  if(T > 5){
    const cols = 99; // 33 * 3
    let sumStd = 0;
    for(let c=0; c<cols; c++){
      let sum=0, sum2=0;
      for(let t=0; t<T; t++){ const v=data[t*cols+c]; sum+=v; sum2+=v*v; }
      const mean=sum/T; const variance=sum2/T-mean*mean;
      sumStd+=Math.sqrt(Math.max(0,variance));
    }
    const avgStd = sumStd/cols;
    if(avgStd >= 0.02) pass('Natural movement', `Temporal variance looks healthy.`);
    else if(avgStd >= 0.005) warn('Very stiff movement', `Is the subject moving enough?`);
    else fail('Frozen skeleton', `No significant movement detected.`);
  }

  // Stats
  let sum=0, cnt=0; for(let i=0; i<data.length; i++) if(!isNaN(data[i]) && isFinite(data[i])){ sum+=data[i]; cnt++; }
  const mean = cnt>0 ? sum/cnt : 0;
  let sumSq=0; for(let i=0; i<data.length; i++) if(!isNaN(data[i]) && isFinite(data[i])) sumSq+=(data[i]-mean)**2;
  const std = cnt>0 ? Math.sqrt(sumSq/cnt) : 0;
  
  let minV=Infinity, maxV=-Infinity;
  for(let i=0; i<data.length; i++){ if(!isNaN(data[i]) && isFinite(data[i])){ if(data[i]<minV) minV=data[i]; if(data[i]>maxV) maxV=data[i]; } }

  const stats = {
    shape: `[${T} × 33 × 3]`,
    frames: T,
    landmarks: 33,
    clips: T >= 48 ? Math.floor((T-48)/16)+1 : (T >= 32 ? 1 : 0),
    mean: mean.toFixed(4),
    std: std.toFixed(4),
    range: `${minV.toFixed(2)} → ${maxV.toFixed(2)}`
  };

  const overall = fails > 0 ? 'fail' : (warns > 0 ? 'warn' : 'pass');
  return { overall, checks, stats, fails, warns };
}

export function createNpyBuffer(frames: number[][][]): ArrayBuffer {
  const T = frames.length;
  // Use [T, 33, 3] shape (Pose only)
  const data = new Float64Array(T * 33 * 3);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < 33; i++) {
      const lm = frames[t][i] || [0, 0, 0];
      data[t * 99 + i * 3 + 0] = lm[0];
      data[t * 99 + i * 3 + 1] = lm[1];
      data[t * 99 + i * 3 + 2] = lm[2];
    }
  }

  const magic = new Uint8Array([147, 78, 85, 77, 80, 89]); // \x93NUMPY
  const dictStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + T + ", 33, 3), }";
  
  let totalLen = 10 + dictStr.length + 1;
  let paddingLen = 64 - (totalLen % 64);
  if (paddingLen === 64) paddingLen = 0;
  const headerStr = dictStr + ' '.repeat(paddingLen) + '\n';
  const headerLen = headerStr.length;
  
  const buffer = new ArrayBuffer(10 + headerLen + data.byteLength);
  const dv = new DataView(buffer);
  
  for (let i = 0; i < 6; i++) dv.setUint8(i, magic[i]);
  dv.setUint8(6, 1);
  dv.setUint8(7, 0);
  dv.setUint16(8, headerLen, true);
  for (let i = 0; i < headerLen; i++) dv.setUint8(10 + i, headerStr.charCodeAt(i));
  
  const outBytes = new Uint8Array(buffer);
  const dataBytes = new Uint8Array(data.buffer);
  outBytes.set(dataBytes, 10 + headerLen);
  
  return buffer;
}
