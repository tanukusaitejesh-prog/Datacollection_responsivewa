export class OneEuroFilter {
  private firstTime = true;
  private xPrev = 0;
  private dxPrev = 0;
  private minCutoff: number;
  private beta: number;
  private dCutoff: number;

  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
  }

  private alpha(cutoff: number, freq: number): number {
    const te = 1.0 / freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  public filter(x: number, freq = 30): number {
    if (this.firstTime) {
      this.firstTime = false;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }

    const dx = (x - this.xPrev) * freq;
    const edx = this.dxPrev + this.alpha(this.dCutoff, freq) * (dx - this.dxPrev);
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const xSmooth = this.xPrev + this.alpha(cutoff, freq) * (x - this.xPrev);

    this.xPrev = xSmooth;
    this.dxPrev = edx;
    return xSmooth;
  }
}
