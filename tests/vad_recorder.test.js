import { describe, it, expect } from 'vitest';
import { SpeechStateMachine } from '../src/vad_recorder.js';

describe('SpeechStateMachine', () => {
  it('emits speech-start when probability crosses threshold', () => {
    const sm = new SpeechStateMachine({ threshold: 0.5, silenceFramesToEnd: 15 });
    expect(sm.feed(0.1)).toEqual({ event: null });
    expect(sm.feed(0.4)).toEqual({ event: null });
    expect(sm.feed(0.51)).toEqual({ event: 'speech-start' });
    expect(sm.feed(0.9)).toEqual({ event: null });
  });

  it('emits speech-end after N silence frames', () => {
    const sm = new SpeechStateMachine({ threshold: 0.5, silenceFramesToEnd: 3 });
    sm.feed(0.9);
    sm.feed(0.1);
    sm.feed(0.1);
    expect(sm.feed(0.1)).toEqual({ event: 'speech-end' });
  });

  it('resets silence counter when speech returns mid-utterance', () => {
    const sm = new SpeechStateMachine({ threshold: 0.5, silenceFramesToEnd: 3 });
    sm.feed(0.9);
    sm.feed(0.1);
    sm.feed(0.6); // reset
    sm.feed(0.1);
    sm.feed(0.1);
    expect(sm.feed(0.1)).toEqual({ event: 'speech-end' });
  });

  it('does not emit speech-start while already speaking', () => {
    const sm = new SpeechStateMachine({ threshold: 0.5, silenceFramesToEnd: 3 });
    expect(sm.feed(0.9)).toEqual({ event: 'speech-start' });
    expect(sm.feed(0.9)).toEqual({ event: null });
  });

  it('reset() clears state back to non-speaking', () => {
    const sm = new SpeechStateMachine({ threshold: 0.5, silenceFramesToEnd: 3 });
    sm.feed(0.9);
    sm.reset();
    expect(sm.feed(0.9)).toEqual({ event: 'speech-start' });
  });
});
