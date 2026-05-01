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

const EXPECTED_LANDMARKS = 33;
const VALUES_PER_LANDMARK = 3;
const VALUES_PER_FRAME = EXPECTED_LANDMARKS * VALUES_PER_LANDMARK;

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function readCoord(frame: number[][] | undefined, landmarkIndex: number, coordIndex: number): number {
  const value = frame?.[landmarkIndex]?.[coordIndex];
  return isFiniteNumber(value) ? value : NaN;
}

export function validatePoseData(frames: number[][][]): ValidationResult {
  const T = frames.length;
  const numLandmarks = frames[0]?.length || 0;
  const data = new Float64Array(T * VALUES_PER_FRAME);

  for (let t = 0; t < T; t++) {
    for (let i = 0; i < EXPECTED_LANDMARKS; i++) {
      const base = t * VALUES_PER_FRAME + i * VALUES_PER_LANDMARK;
      data[base + 0] = readCoord(frames[t], i, 0);
      data[base + 1] = readCoord(frames[t], i, 1);
      data[base + 2] = readCoord(frames[t], i, 2);
    }
  }

  const checks: ValidationResult['checks'] = [];
  let fails = 0;
  let warns = 0;

  const pass = (name: string, detail: string) => checks.push({ s: 'pass', name, detail });
  const fail = (name: string, detail: string) => { checks.push({ s: 'fail', name, detail }); fails++; };
  const warn = (name: string, detail: string) => { checks.push({ s: 'warn', name, detail }); warns++; };

  const invalidFrameCount = frames.filter(frame => frame.length !== EXPECTED_LANDMARKS).length;
  if (T > 0 && invalidFrameCount === 0) {
    pass('Landmarks count is 33', 'Detected exactly 33 MediaPipe landmarks per frame.');
  } else {
    fail('Invalid landmarks count', `Expected 33 per frame, got ${numLandmarks || 0} in the first frame.`);
  }

  if (T >= 48) pass('Frames >= 48 (WINDOW_SIZE)', `${T} frames -> ${Math.floor((T - 48) / 16) + 1} full clips`);
  else if (T >= 32) warn('Frames >= 32 but < 48', `${T} frames - will be padded.`);
  else fail('Too few frames (< 32)', `Only ${T} frames. Minimum is 32.`);

  let nanCount = 0;
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) nanCount++;
  }
  if (nanCount === 0) pass('No invalid numbers', 'All values are finite numbers.');
  else if (data.length > 0 && nanCount / data.length < 0.05) warn(`${nanCount} invalid values`, 'Small amount detected.');
  else fail(`${nanCount} invalid values`, 'Too many missing landmarks.');

  let missingLandmarks = 0;
  let totalLandmarks = 0;
  let outsideXY = 0;
  let totalXY = 0;
  const frameAreas: number[] = [];

  for (let t = 0; t < T; t++) {
    const xs: number[] = [];
    const ys: number[] = [];

    for (let i = 0; i < EXPECTED_LANDMARKS; i++) {
      const x = readCoord(frames[t], i, 0);
      const y = readCoord(frames[t], i, 1);
      const z = readCoord(frames[t], i, 2);
      totalLandmarks++;

      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) || (x === 0 && y === 0 && z === 0)) {
        missingLandmarks++;
      }

      if (Number.isFinite(x) && Number.isFinite(y)) {
        totalXY += 2;
        if (x < -0.2 || x > 1.2) outsideXY++;
        if (y < -0.2 || y > 1.2) outsideXY++;
        if (x >= -0.2 && x <= 1.2 && y >= -0.2 && y <= 1.2) {
          xs.push(x);
          ys.push(y);
        }
      }
    }

    if (xs.length > 1 && ys.length > 1) {
      frameAreas.push((Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)));
    }
  }

  const missingPct = totalLandmarks > 0 ? missingLandmarks / totalLandmarks : 1;
  if (missingPct < 0.05) pass('No padded pose landmarks', `${(missingPct * 100).toFixed(1)}% all-zero landmarks`);
  else if (missingPct < 0.25) warn('Some padded pose landmarks', `${(missingPct * 100).toFixed(1)}% all-zero landmarks`);
  else fail('Too many missing pose landmarks', `${(missingPct * 100).toFixed(1)}% all-zero landmarks`);

  const outsidePct = totalXY > 0 ? outsideXY / totalXY : 1;
  if (outsidePct < 0.1) pass('Coordinates stay near camera frame', `${(outsidePct * 100).toFixed(1)}% x/y values outside tolerance`);
  else if (outsidePct < 0.35) warn('Partial out-of-frame motion', `${(outsidePct * 100).toFixed(1)}% x/y values outside tolerance`);
  else fail('Unstable pose geometry', `${(outsidePct * 100).toFixed(1)}% x/y values outside tolerance`);

  const avgArea = frameAreas.length > 0 ? frameAreas.reduce((sum, area) => sum + area, 0) / frameAreas.length : 0;
  if (avgArea >= 0.004) pass('Subject scale is trackable', `Average pose box area ${avgArea.toFixed(4)}`);
  else if (avgArea >= 0.001) warn('Small subject in frame', `Average pose box area ${avgArea.toFixed(4)}; usable but inspect if needed.`);
  else fail('Subject too small to trust', `Average pose box area ${avgArea.toFixed(4)}.`);

  if (T > 5) {
    let sumStd = 0;
    let stdCols = 0;
    for (let c = 0; c < VALUES_PER_FRAME; c++) {
      let sum = 0;
      let sum2 = 0;
      let count = 0;
      for (let t = 0; t < T; t++) {
        const v = data[t * VALUES_PER_FRAME + c];
        if (Number.isFinite(v)) {
          sum += v;
          sum2 += v * v;
          count++;
        }
      }
      if (count > 1) {
        const mean = sum / count;
        const variance = sum2 / count - mean * mean;
        sumStd += Math.sqrt(Math.max(0, variance));
        stdCols++;
      }
    }
    const avgStd = stdCols > 0 ? sumStd / stdCols : 0;

    let totalDelta = 0;
    let deltaCount = 0;
    for (let t = 1; t < T; t++) {
      for (let c = 0; c < VALUES_PER_FRAME; c++) {
        const prev = data[(t - 1) * VALUES_PER_FRAME + c];
        const next = data[t * VALUES_PER_FRAME + c];
        if (Number.isFinite(prev) && Number.isFinite(next)) {
          totalDelta += Math.abs(next - prev);
          deltaCount++;
        }
      }
    }
    const meanDelta = deltaCount > 0 ? totalDelta / deltaCount : 0;

    if (avgStd >= 0.02) pass('Natural movement', 'Temporal variance looks healthy.');
    else if (meanDelta < 0.00001) fail('Repeated identical frames', 'No frame-to-frame landmark change was detected.');
    else if (avgStd >= 0.002) warn('Limited movement', 'Usable for static or subtle actions; record longer if this should be dynamic.');
    else warn('Minimal movement', 'Very little motion detected; acceptable only for a static action.');
  }

  let sum = 0;
  let cnt = 0;
  for (let i = 0; i < data.length; i++) {
    if (Number.isFinite(data[i])) {
      sum += data[i];
      cnt++;
    }
  }
  const mean = cnt > 0 ? sum / cnt : 0;
  let sumSq = 0;
  for (let i = 0; i < data.length; i++) {
    if (Number.isFinite(data[i])) sumSq += (data[i] - mean) ** 2;
  }
  const std = cnt > 0 ? Math.sqrt(sumSq / cnt) : 0;

  let minV = Infinity;
  let maxV = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (Number.isFinite(data[i])) {
      if (data[i] < minV) minV = data[i];
      if (data[i] > maxV) maxV = data[i];
    }
  }

  const stats = {
    shape: `[${T} x ${EXPECTED_LANDMARKS} x ${VALUES_PER_LANDMARK}]`,
    frames: T,
    landmarks: EXPECTED_LANDMARKS,
    clips: T >= 48 ? Math.floor((T - 48) / 16) + 1 : (T >= 32 ? 1 : 0),
    mean: mean.toFixed(4),
    std: std.toFixed(4),
    range: cnt > 0 ? `${minV.toFixed(2)} -> ${maxV.toFixed(2)}` : 'n/a'
  };

  const overall = fails > 0 ? 'fail' : (warns > 0 ? 'warn' : 'pass');
  return { overall, checks, stats, fails, warns };
}

export function createNpyBuffer(frames: number[][][]): ArrayBuffer {
  const T = frames.length;
  const data = new Float64Array(T * VALUES_PER_FRAME);

  for (let t = 0; t < T; t++) {
    for (let i = 0; i < EXPECTED_LANDMARKS; i++) {
      const base = t * VALUES_PER_FRAME + i * VALUES_PER_LANDMARK;
      const x = frames[t]?.[i]?.[0];
      const y = frames[t]?.[i]?.[1];
      const z = frames[t]?.[i]?.[2];
      data[base + 0] = isFiniteNumber(x) ? x : Number.NaN;
      data[base + 1] = isFiniteNumber(y) ? y : Number.NaN;
      data[base + 2] = isFiniteNumber(z) ? z : Number.NaN;
    }
  }

  const magic = new Uint8Array([147, 78, 85, 77, 80, 89]);
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
