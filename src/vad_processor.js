/**
 * AudioWorklet: buffers raw mic PCM at the AudioContext sample rate (we set
 * 16kHz in the recorder) into 512-sample frames and posts each completed
 * frame to the main thread. Browser delivers 128-sample quanta, so 4 quanta
 * (~32ms at 16kHz) fill one frame.
 */
class VADProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(512);
    this.idx = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buffer[this.idx++] = ch[i];
      if (this.idx >= 512) {
        this.port.postMessage(this.buffer.slice());
        this.idx = 0;
      }
    }
    return true;
  }
}

registerProcessor('vad-processor', VADProcessor);
