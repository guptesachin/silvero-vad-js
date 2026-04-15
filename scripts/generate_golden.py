"""Run ONNX Runtime over 10 deterministic 512-sample frames; save inputs + outputs.

The VAD model keeps LSTM state across frames within an utterance. We thread the
state forward across all 10 frames; the golden fixture records the input, the
state-in at each step, the resulting speech probability, and the state-out.

Output: tests/fixtures/golden.json
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = "weights/silero_vad_v5.onnx"
OUT_PATH = "tests/fixtures/golden.json"
SR = 16000
WIN = 512
STATE_SHAPE = (2, 1, 128)


def make_frames() -> list[np.ndarray]:
    rng = np.random.default_rng(seed=42)
    frames = []
    # 0-2: pure silence
    for _ in range(3):
        frames.append(np.zeros(WIN, dtype=np.float32))
    # 3-5: low-amplitude white noise
    for _ in range(3):
        frames.append((rng.standard_normal(WIN) * 0.01).astype(np.float32))
    # 6-9: 200 Hz sine at 0.3 amplitude (continuous across frames)
    t = np.arange(4 * WIN) / SR
    sine = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for k in range(4):
        frames.append(sine[k * WIN:(k + 1) * WIN].copy())
    return frames


def main() -> None:
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    assert set(input_names) == {"input", "state", "sr"}, f"unexpected inputs: {input_names}"

    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)
    out = {"sr": SR, "win": WIN, "tolerance": 1e-4, "frames": []}

    for frame in make_frames():
        x = frame.reshape(1, WIN).astype(np.float32)
        state_in = state.copy()
        outputs = sess.run(None, {"input": x, "state": state, "sr": sr})
        prob, new_state = outputs[0], outputs[1]
        out["frames"].append({
            "input": frame.tolist(),
            "state_in": state_in.flatten().tolist(),
            "speech_prob": float(prob.flatten()[0]),
            "state_out": new_state.flatten().tolist(),
        })
        state = new_state

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(out))
    probs = [f["speech_prob"] for f in out["frames"]]
    print(f"Wrote {len(out['frames'])} frames to {OUT_PATH}")
    print(f"Speech probabilities: {[f'{p:.4f}' for p in probs]}")


if __name__ == "__main__":
    main()
