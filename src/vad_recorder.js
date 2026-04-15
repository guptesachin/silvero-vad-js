import { SileroVADJS } from './silero_vad.js';
import { loadWeights } from './weights_loader.js';

/**
 * Pure state machine: takes a frame's speech probability, returns whether a
 * speech-start or speech-end event happened. No audio, no Web Audio API,
 * no DOM — fully unit-testable.
 */
export class SpeechStateMachine {
  constructor({ threshold = 0.5, silenceFramesToEnd = 15 } = {}) {
    this.threshold = threshold;
    this.silenceFramesToEnd = silenceFramesToEnd;
    this.speaking = false;
    this.silenceCount = 0;
  }

  feed(prob) {
    if (prob >= this.threshold) {
      this.silenceCount = 0;
      if (!this.speaking) {
        this.speaking = true;
        return { event: 'speech-start' };
      }
      return { event: null };
    }
    if (this.speaking) {
      this.silenceCount++;
      if (this.silenceCount >= this.silenceFramesToEnd) {
        this.speaking = false;
        this.silenceCount = 0;
        return { event: 'speech-end' };
      }
    }
    return { event: null };
  }

  reset() {
    this.speaking = false;
    this.silenceCount = 0;
  }
}

function concatFloat32(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/**
 * Browser-only: wires getUserMedia → AudioContext(16000) → AudioWorklet → VAD,
 * dispatching 'speech-start' and 'speech-end' events (the latter carries the
 * accumulated audio chunks in event.detail.audio as a Float32Array).
 */
export class VADRecorder extends EventTarget {
  constructor({
    weightsBinUrl,
    weightsManifestUrl,
    workletUrl,
    threshold = 0.5,
    silenceMs = 480, // 480ms ≈ 15 frames at 32ms/frame
  } = {}) {
    super();
    this.weightsBinUrl = weightsBinUrl;
    this.weightsManifestUrl = weightsManifestUrl;
    this.workletUrl = workletUrl;
    const silenceFramesToEnd = Math.max(1, Math.round(silenceMs / 32));
    this.stateMachine = new SpeechStateMachine({ threshold, silenceFramesToEnd });
    this.vad = null;
    this.audioContext = null;
    this.stream = null;
    this.audioChunks = [];
  }

  async start() {
    const weights = await loadWeights(this.weightsBinUrl, this.weightsManifestUrl);
    this.vad = new SileroVADJS(weights);

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    await this.audioContext.audioWorklet.addModule(this.workletUrl);

    const source = this.audioContext.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.audioContext, 'vad-processor');
    node.port.onmessage = (e) => this._onFrame(e.data);
    source.connect(node);
    // Intentionally not connecting node to destination — we don't want mic passthrough.
  }

  _onFrame(frame) {
    const prob = this.vad.process(frame);
    const { event } = this.stateMachine.feed(prob);
    if (event === 'speech-start') {
      this.audioChunks = [frame];
      this.dispatchEvent(new CustomEvent('speech-start', { detail: { prob } }));
    } else if (this.stateMachine.speaking) {
      this.audioChunks.push(frame);
    } else if (event === 'speech-end') {
      this.dispatchEvent(new CustomEvent('speech-end', {
        detail: { audio: concatFloat32(this.audioChunks), sampleRate: 16000 },
      }));
      this.audioChunks = [];
      this.vad.reset();
    }
    // Always surface the raw probability for UIs that want a live meter.
    this.dispatchEvent(new CustomEvent('frame', { detail: { prob } }));
  }

  async stop() {
    if (this.audioContext) await this.audioContext.close();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.audioContext = null;
    this.stream = null;
  }
}
