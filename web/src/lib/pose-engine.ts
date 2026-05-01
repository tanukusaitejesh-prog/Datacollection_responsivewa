import { 
  PoseLandmarker, FaceLandmarker, HandLandmarker, 
  FilesetResolver, 
  PoseLandmarkerResult, FaceLandmarkerResult, HandLandmarkerResult 
} from '@mediapipe/tasks-vision';

export type HolisticCallback = (results: {
  pose: PoseLandmarkerResult | null;
  face: FaceLandmarkerResult | null;
  hands: HandLandmarkerResult | null;
}) => void;

const SKELETON_EDGES: Array<[number, number]> = [
  // Face
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
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

const DRAW_VISIBILITY_THRESHOLD = 0.12;

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

function landmarkVisibility(lm: any): number {
  return Number.isFinite(lm?.visibility) ? lm.visibility : 0;
}

function isDrawableLandmark(lm: any): boolean {
  return (
    Number.isFinite(lm?.x) &&
    Number.isFinite(lm?.y) &&
    Number.isFinite(lm?.z) &&
    landmarkVisibility(lm) >= DRAW_VISIBILITY_THRESHOLD &&
    lm.x >= -0.2 &&
    lm.x <= 1.2 &&
    lm.y >= -0.2 &&
    lm.y <= 1.2
  );
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
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.35,
        minPosePresenceConfidence: 0.45,
        minTrackingConfidence: 0.5
      });

      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numFaces: 1,
        minFaceDetectionConfidence: 0.25,
        minFacePresenceConfidence: 0.25,
        minTrackingConfidence: 0.25,
        outputFaceBlendshapes: true
      });

      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.25,
        minHandPresenceConfidence: 0.25,
        minTrackingConfidence: 0.25
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

      // Draw Edges
      SKELETON_EDGES.forEach(([from, to]) => {
        const p1 = landmarks[from];
        const p2 = landmarks[to];
        if (p1 && p2 && isDrawableLandmark(p1) && isDrawableLandmark(p2)) {
          const alpha = clamp((landmarkVisibility(p1) + landmarkVisibility(p2)) * 0.5, 0.2, 1);
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.strokeStyle = `rgba(120, 220, 255, ${alpha})`;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      });

      // Draw All Joints
      landmarks.forEach((lm) => {
        if (!isDrawableLandmark(lm)) return;

        const x = lm.x * width;
        const y = lm.y * height;
        const radius = 2 + landmarkVisibility(lm) * 3;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = zToColor(lm.z);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }

    // Quick draw face points (small dots)
    if (results.face && results.face.faceLandmarks && results.face.faceLandmarks.length > 0) {
      const faceLms = results.face.faceLandmarks[0];
      ctx.fillStyle = 'rgba(0, 255, 0, 0.6)';
      faceLms.forEach(lm => {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 1, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // Quick draw hand points
    if (results.hands && results.hands.landmarks && results.hands.landmarks.length > 0) {
      ctx.fillStyle = 'rgba(255, 100, 0, 0.8)';
      results.hands.landmarks.forEach(handLms => {
        handLms.forEach(lm => {
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, 2, 0, 2 * Math.PI);
          ctx.fill();
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
