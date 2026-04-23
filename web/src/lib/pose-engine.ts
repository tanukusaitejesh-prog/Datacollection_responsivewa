import { PoseLandmarker, FilesetResolver, PoseLandmarkerResult } from '@mediapipe/tasks-vision';

export type PoseCallback = (results: PoseLandmarkerResult) => void;

const SKELETON_EDGES: Array<[number, number]> = [
  [11, 13], [13, 15], [12, 14], [14, 16], [11, 12], [23, 24],
  [11, 23], [12, 24], [23, 25], [25, 27], [24, 26], [26, 28]
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
  private landmarker: PoseLandmarker | null = null;
  private canvasElement: HTMLCanvasElement;
  private canvasCtx: CanvasRenderingContext2D;
  private onResultsCallbacks: PoseCallback[] = [];
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
      
      this.landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
      });

      this.isLoaded = true;
      console.log("Pose Landmarker initialized");
    } catch (err) {
      console.error("Failed to initialize Pose Landmarker:", err);
    }
  }

  public onResults(cb: PoseCallback) {
    this.onResultsCallbacks.push(cb);
  }

  public async send(video: HTMLVideoElement) {
    if (!this.landmarker || !this.isLoaded) return;
    const startTimeMs = performance.now();
    const result = this.landmarker.detectForVideo(video, startTimeMs);
    this.handleResults(result);
  }

  private handleResults(results: PoseLandmarkerResult) {
    const ctx = this.canvasCtx;
    const { width, height } = this.canvasElement;

    ctx.clearRect(0, 0, width, height);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];

      // Draw Edges
      SKELETON_EDGES.forEach(([from, to]) => {
        const p1 = landmarks[from];
        const p2 = landmarks[to];
        if (p1 && p2) {
          const alpha = clamp(((p1.visibility ?? 0) + (p2.visibility ?? 0)) * 0.5, 0.15, 1);
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.strokeStyle = `rgba(120, 220, 255, ${alpha})`;
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      });

      // Draw Joints
      landmarks.forEach((lm, idx) => {
        // Only draw major joints to keep it clean (same indices as edges + hips)
        const majorIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
        if (!majorIndices.includes(idx)) return;

        const x = lm.x * width;
        const y = lm.y * height;
        const radius = 2 + (lm.visibility ?? 0) * 3;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = zToColor(lm.z);
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
    }

    this.onResultsCallbacks.forEach(cb => cb(results));
  }

  public close() {
    this.landmarker?.close();
  }
}
