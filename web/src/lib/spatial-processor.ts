/**
 * SpatialProcessor replicates the screening app's logic for 100% compatible NPY files.
 * Handles: Interpolation, Root Centering, Orientation Alignment, Scale Normalization, and Smoothing.
 */
export class SpatialProcessor {
  private confidenceThreshold: number;

  private readonly L_HIP = 23;
  private readonly R_HIP = 24;
  private readonly L_SHOULDER = 11;
  private readonly R_SHOULDER = 12;

  constructor(confidenceThreshold = 0.5) {
    this.confidenceThreshold = confidenceThreshold;
  }

  /**
   * Processes a sequence of landmarks.
   * @param landmarks np.array of shape [T, 33, 4] (X, Y, Z, Visibility)
   * @returns np.array of shape [T, 33, 3] (X, Y, Z)
   */
  public processSequence(landmarks: number[][][]): number[][][] {
    const T = landmarks.length;
    if (T === 0) return [];

    // 1. Interpolate missing joints (Linear)
    let processed = this.interpolate(landmarks);

    // 2. Root Centering (Pelvis midpoint)
    processed = this.rootCenter(processed);

    // 3. Orientation Alignment (XY Plane Rotation to point "Up")
    processed = this.alignOrientation(processed);

    // 4. Scale Normalization (Median Torso Length)
    processed = this.normalizeScale(processed);

    // 5. Temporal Smoothing (Savitzky-Golay approximation)
    processed = this.smooth(processed);

    return processed;
  }

  private interpolate(landmarks: number[][][]): number[][][] {
    const T = landmarks.length;
    const numJoints = 33;
    const out: number[][][] = Array.from({ length: T }, () =>
      Array.from({ length: numJoints }, () => [0, 0, 0])
    );

    for (let j = 0; j < numJoints; j++) {
      // Extract trajectory for joint j
      const trajectory: (number[] | null)[] = landmarks.map((frame) => {
        const lm = frame[j] || [0, 0, 0, 0];
        const visibility = lm[3] !== undefined ? lm[3] : 1.0; // Default to 1 if missing
        return visibility >= this.confidenceThreshold
          ? [lm[0], lm[1], lm[2]]
          : null;
      });

      // Linear interpolation per dimension
      for (let dim = 0; dim < 3; dim++) {
        let lastValidIdx = -1;
        for (let t = 0; t < T; t++) {
          if (trajectory[t] !== null) {
            out[t][j][dim] = trajectory[t]![dim];
            if (lastValidIdx !== -1 && t - lastValidIdx > 1) {
              const startVal = trajectory[lastValidIdx]![dim];
              const endVal = trajectory[t]![dim];
              for (let k = lastValidIdx + 1; k < t; k++) {
                const alpha = (k - lastValidIdx) / (t - lastValidIdx);
                out[k][j][dim] = startVal + alpha * (endVal - startVal);
              }
            }
            lastValidIdx = t;
          }
        }

        // Handle edge cases
        if (lastValidIdx !== -1) {
          let firstValidIdx = -1;
          for (let i = 0; i < T; i++) {
            if (trajectory[i] !== null) {
              firstValidIdx = i;
              break;
            }
          }
          if (firstValidIdx !== -1) {
            // Fill before first valid
            for (let i = 0; i < firstValidIdx; i++)
              out[i][j][dim] = trajectory[firstValidIdx]![dim];
            // Fill after last valid
            for (let i = lastValidIdx + 1; i < T; i++)
              out[i][j][dim] = trajectory[lastValidIdx]![dim];
          }
        }
      }
    }
    return out;
  }

  private rootCenter(landmarks: number[][][]): number[][][] {
    return landmarks.map((frame) => {
      const lHip = frame[this.L_HIP];
      const rHip = frame[this.R_HIP];
      const root = [
        (lHip[0] + rHip[0]) / 2.0,
        (lHip[1] + rHip[1]) / 2.0,
        (lHip[2] + rHip[2]) / 2.0,
      ];
      return frame.map((lm) => [
        lm[0] - root[0],
        lm[1] - root[1],
        lm[2] - root[2],
      ]);
    });
  }

  private alignOrientation(landmarks: number[][][]): number[][][] {
    const T = landmarks.length;
    let avgTorsoX = 0;
    let avgTorsoY = 0;

    for (let t = 0; t < T; t++) {
      const root2d = [
        (landmarks[t][this.L_HIP][0] + landmarks[t][this.R_HIP][0]) / 2.0,
        (landmarks[t][this.L_HIP][1] + landmarks[t][this.R_HIP][1]) / 2.0,
      ];
      const shld2d = [
        (landmarks[t][this.L_SHOULDER][0] + landmarks[t][this.R_SHOULDER][0]) /
          2.0,
        (landmarks[t][this.L_SHOULDER][1] + landmarks[t][this.R_SHOULDER][1]) /
          2.0,
      ];
      avgTorsoX += shld2d[0] - root2d[0];
      avgTorsoY += shld2d[1] - root2d[1];
    }
    avgTorsoX /= T;
    avgTorsoY /= T;

    // Rotate to point Up (-90 degrees in image space)
    const dTheta = -Math.PI / 2.0 - Math.atan2(avgTorsoY, avgTorsoX);
    const cosTr = Math.cos(dTheta);
    const sinTr = Math.sin(dTheta);

    return landmarks.map((frame) =>
      frame.map((lm) => {
        const x = lm[0];
        const y = lm[1];
        return [x * cosTr - y * sinTr, x * sinTr + y * cosTr, lm[2]];
      })
    );
  }

  private normalizeScale(landmarks: number[][][]): number[][][] {
    const T = landmarks.length;
    const torsoLens: number[] = [];

    for (let t = 0; t < T; t++) {
      const lHip = landmarks[t][this.L_HIP];
      const rHip = landmarks[t][this.R_HIP];
      const lShld = landmarks[t][this.L_SHOULDER];
      const rShld = landmarks[t][this.R_SHOULDER];

      const root3d = [
        (lHip[0] + rHip[0]) / 2.0,
        (lHip[1] + rHip[1]) / 2.0,
        (lHip[2] + rHip[2]) / 2.0,
      ];
      const shld3d = [
        (lShld[0] + rShld[0]) / 2.0,
        (lShld[1] + rShld[1]) / 2.0,
        (lShld[2] + rShld[2]) / 2.0,
      ];

      const dist = Math.sqrt(
        Math.pow(shld3d[0] - root3d[0], 2) +
          Math.pow(shld3d[1] - root3d[1], 2) +
          Math.pow(shld3d[2] - root3d[2], 2)
      );
      torsoLens.push(dist);
    }

    torsoLens.sort((a, b) => a - b);
    const medianTorsoLen = torsoLens[Math.floor(T / 2)];
    const scale = medianTorsoLen > 1e-5 ? 1.0 / medianTorsoLen : 1.0;

    return landmarks.map((frame) =>
      frame.map((lm) => [lm[0] * scale, lm[1] * scale, lm[2] * scale])
    );
  }

  private smooth(landmarks: number[][][]): number[][][] {
    const T = landmarks.length;
    if (T < 5) return landmarks;

    // Savitzky-Golay (Window=5, Poly=2) coefficients for the central point:
    // [-3, 12, 17, 12, -3] / 35
    const coeffs = [-3 / 35, 12 / 35, 17 / 35, 12 / 35, -3 / 35];
    const out: number[][][] = JSON.parse(JSON.stringify(landmarks));

    for (let t = 2; t < T - 2; t++) {
      for (let j = 0; j < 33; j++) {
        for (let dim = 0; dim < 3; dim++) {
          let smoothed = 0;
          for (let k = 0; k < 5; k++) {
            smoothed += landmarks[t - 2 + k][j][dim] * coeffs[k];
          }
          out[t][j][dim] = smoothed;
        }
      }
    }
    return out;
  }
}
