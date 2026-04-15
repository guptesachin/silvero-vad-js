"""Compare JS probabilities on real mic audio against the OFFICIAL SileroVAD
wrapper (which prepends 64 samples of context from the previous frame).
"""
import json
import numpy as np
import torch
from silero_vad import load_silero_vad

d = json.load(open("captured_frames.json"))
frames = d["frames"]
print(f"Loaded {len(frames)} frames")

model = load_silero_vad(onnx=True)
# Feed each 512-sample frame through the wrapper; it handles context internally.
official_probs = []
for f in frames:
    x = torch.from_numpy(np.array(f["input"], dtype=np.float32))
    out = model(x, 16000)
    official_probs.append(float(out.item()))

js_probs = [f["prob"] for f in frames]
amps = [float(np.abs(f["input"]).max()) for f in frames]

print(f"{'frame':>5} {'amp':>8} {'js_prob':>10} {'official':>10} {'diff':>10}")
for i in range(len(frames)):
    if official_probs[i] > 0.1 or js_probs[i] > 0.1 or amps[i] > 0.3 or i < 5 or i % 20 == 0:
        print(f"{i:>5d} {amps[i]:>8.4f} {js_probs[i]:>10.6f} {official_probs[i]:>10.6f} {js_probs[i]-official_probs[i]:>+10.6f}")

print(f"\nJS:       min={min(js_probs):.4f}  max={max(js_probs):.4f}  mean={sum(js_probs)/len(js_probs):.4f}")
print(f"Official: min={min(official_probs):.4f}  max={max(official_probs):.4f}  mean={sum(official_probs)/len(official_probs):.4f}")
print(f"Frames with official > 0.5: {sum(1 for p in official_probs if p > 0.5)}")
