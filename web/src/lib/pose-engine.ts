import { 
  PoseLandmarker, FaceLandmarker, HandLandmarker, 
  FilesetResolver, 
  PoseLandmarkerResult, FaceLandmarkerResult, HandLandmarkerResult 
} from '@mediapipe/tasks-vision';
// Filtering removed from preview for absolute zero-latency

export type HolisticCallback = (results: {
  pose: PoseLandmarkerResult | null;
  face: FaceLandmarkerResult | null;
  hands: HandLandmarkerResult | null;
}) => void;

const SKELETON_EDGES: Array<[number, number]> = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  // Head to Body
  [0, 11], [0, 12],
  // Torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Arms
  [11, 13], [13, 15], [12, 14], [14, 16],
  // Hands
  [15, 17], [15, 19], [15, 21], [17, 19], 
  [16, 18], [16, 20], [16, 22], [18, 20],
  // Legs & Feet
  [23, 25], [25, 27], [27, 29], [29, 31], [31, 27],
  [24, 26], [26, 28], [28, 30], [30, 32], [32, 28]
];

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17] // Palm base
];

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function zToColor(z: number): string {
  const t = clamp((z + 0.45) / 0.9, 0, 1);
  const red = Math.round(255 * (1 - t));
  const blue = Math.round(255 * t);
  return `rgb(${red},180,${blue})`;
}

export class PoseEngine {
  private poseLandmarker: PoseLandmarker | null = null;
  private faceLandmarker: FaceLandmarker | null = null;
  private handLandmarker: HandLandmarker | null = null;
  
  private canvasElement: HTMLCanvasElement;
  private canvasCtx: CanvasRenderingContext2D;
  private onResultsCallbacks: HolisticCallback[] = [];
  private isLoaded = false;

  constructor(_video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    this.canvasElement = canvas;
    this.canvasCtx = canvas.getContext('2d')!;
    this.init();
  }

  private async init() {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      
      this.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.4,
        minFacePresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
        outputFaceBlendshapes: true
      });

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.4
      });

      this.isLoaded = true;
      console.log("Pose Landmarker initialized");
    } catch (err) {
      console.error("Failed to initialize Pose Landmarker:", err);
    }
  }

  public onResults(cb: HolisticCallback) {
    this.onResultsCallbacks.push(cb);
  }

  public async send(video: HTMLVideoElement, options: { face: boolean; hands: boolean } = { face: true, hands: true }) {
    if (!this.poseLandmarker || !this.isLoaded) return;
    const startTimeMs = performance.now();
    
    // MediaPipe detectForVideo is synchronous, so we don't need Promise.all
    // but we can still run them conditionally
    const poseResult = this.poseLandmarker.detectForVideo(video, startTimeMs);
    
    let faceResult: FaceLandmarkerResult | null = null;
    if (options.face && this.faceLandmarker) {
      faceResult = this.faceLandmarker.detectForVideo(video, startTimeMs);
    }

    let handResult: HandLandmarkerResult | null = null;
    if (options.hands && this.handLandmarker) {
      handResult = this.handLandmarker.detectForVideo(video, startTimeMs);
    }
    
    this.handleResults({ pose: poseResult, face: faceResult, hands: handResult });
  }

  private handleResults(results: {
    pose: PoseLandmarkerResult | null;
    face: FaceLandmarkerResult | null;
    hands: HandLandmarkerResult | null;
  }) {
    const ctx = this.canvasCtx;
    const { width, height } = this.canvasElement;

    ctx.clearRect(0, 0, width, height);

    if (results.pose && results.pose.landmarks && results.pose.landmarks.length > 0) {
      const landmarks = results.pose.landmarks[0];
      
      // Calculate synthetic neck point (midpoint of shoulders)
      const p11 = landmarks[11];
      const p12 = landmarks[12];
      const nose = landmarks[0];
      
      let neckX = 0, neckY = 0;
      let hasNeck = false;
      if (p11 && p12) {
        neckX = (p11.x + p12.x) / 2;
        neckY = (p11.y + p12.y) / 2;
        hasNeck = true;
      }

      // Pre-calculate pixel coordinates and apply foot spread
      const pixelCoords = landmarks.map((lm, i) => {
        let x = lm.x * width;
        let y = lm.y * height;
        
        // Foot landmark spread (indices 29, 30, 31, 32)
        // 27=L_Ankle, 28=R_Ankle, 29=L_Heel, 30=R_Heel, 31=L_FootIndex, 32=R_FootIndex
        if (i >= 29 && i <= 32) {
          const ankleIdx = i % 2 === 1 ? 27 : 28; // 29,31 -> 27; 30,32 -> 28
          const ankle = landmarks[ankleIdx];
          if (ankle) {
            const ax = ankle.x * width;
            const ay = ankle.y * height;
            const dx = x - ax;
            const dy = y - ay;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 5) {
              // Apply minimum spread of 5px
              const scale = 5 / (dist || 1);
              x = ax + dx * scale;
              y = ay + dy * scale;
            }
          }
        }
        
        return { x, y, z: lm.z, visibility: lm.visibility ?? 0 };
      });

      // Draw Edges (Performance Optimized)
      SKELETON_EDGES.forEach(([from, to]) => {
        const p1 = pixelCoords[from];
        const p2 = pixelCoords[to];
        if (p1 && p2 && p1.visibility > 0.4 && p2.visibility > 0.4) {
          const alpha = clamp((p1.visibility + p2.visibility) * 0.5, 0.15, 1);
          ctx.beginPath();
          ctx.moveTo(Math.round(p1.x), Math.round(p1.y));
          ctx.lineTo(Math.round(p2.x), Math.round(p2.y));
          ctx.strokeStyle = `rgba(120, 220, 255, ${alpha})`;
          ctx.lineWidth = 4;
          ctx.stroke();
        }
      });

      // Draw Synthetic Neck Connection
      if (hasNeck && nose && nose.visibility > 0.4) {
        const nx = Math.round(nose.x * width);
        const ny = Math.round(nose.y * height);
        const nkx = Math.round(neckX * width);
        const nky = Math.round(neckY * height);
        
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(nkx, nky);
        ctx.strokeStyle = `rgba(120, 220, 255, ${nose.visibility})`;
        ctx.lineWidth = 4;
        ctx.stroke();
      }

      // Draw All Joints
      pixelCoords.forEach((lm) => {
        if (lm.visibility < 0.4) return;
        
        const x = Math.round(lm.x);
        const y = Math.round(lm.y);
        const radius = 2 + lm.visibility * 3;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = zToColor(lm.z);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }

    // Detailed Face Mesh (Contours)
    if (results.face && results.face.faceLandmarks && results.face.faceLandmarks.length > 0) {
      const faceLms = results.face.faceLandmarks[0];
      
      // Draw main face points with small subtle dots
      ctx.fillStyle = 'rgba(0, 255, 255, 0.4)';
      faceLms.forEach((lm, i) => {
        if (i % 4 === 0) { // Optimize: only draw every 4th point for mesh
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, 0.8, 0, 2 * Math.PI);
          ctx.fill();
        }
      });

      // Draw Lip & Eye Contours for "Detailed AF" look
      const lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61];
      const leftEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 33];
      const rightEye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 263];

      ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
      ctx.lineWidth = 1;

      const drawPath = (indices: number[]) => {
        ctx.beginPath();
        indices.forEach((idx, i) => {
          const pt = faceLms[idx];
          if (pt) {
            if (i === 0) ctx.moveTo(pt.x * width, pt.y * height);
            else ctx.lineTo(pt.x * width, pt.y * height);
          }
        });
        ctx.stroke();
      };

      drawPath(lips);
      drawPath(leftEye);
      drawPath(rightEye);
    }

    // Detailed Hand Skeletons
    if (results.hands && results.hands.landmarks && results.hands.landmarks.length > 0) {
      results.hands.landmarks.forEach(handLms => {
        // Draw connections
        ctx.strokeStyle = 'rgba(255, 180, 0, 0.9)';
        ctx.lineWidth = 2;
        HAND_CONNECTIONS.forEach(([from, to]) => {
          const p1 = handLms[from];
          const p2 = handLms[to];
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x * width, p1.y * height);
            ctx.lineTo(p2.x * width, p2.y * height);
            ctx.stroke();
          }
        });

        // Draw points
        ctx.fillStyle = 'white';
        handLms.forEach(lm => {
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, 2.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#FF8C00';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      });
    }

    this.onResultsCallbacks.forEach(cb => cb(results));
  }

  public close() {
    this.poseLandmarker?.close();
    this.faceLandmarker?.close();
    this.handLandmarker?.close();
  }
}
