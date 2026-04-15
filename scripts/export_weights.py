"""Export the SileroVAD v5 16kHz-branch weights to a flat float32 binary + manifest.

v5 stores weights as Constant nodes inside the top-level If(sr==16000) subgraph,
not as graph.initializer. We walk only the then_branch (16kHz) and extract the
meaningful float32 constants under their logical names (stripping the ONNX
inlining prefix).

Output:
  weights/silero_vad_v5.bin            # raw little-endian float32 concatenation
  weights/silero_vad_v5.manifest.json  # { "tensors": [{name, shape, offset, length}, ...] }

Offsets and lengths are in *float elements* (JS loader multiplies by 4 for bytes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# Only these logical names end up in the JS runtime.
WANTED = {
    "stft.forward_basis_buffer",
    "encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias",
    "encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias",
    "encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias",
    "encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias",
    "decoder.rnn.weight_ih", "decoder.rnn.weight_hh",
    "decoder.rnn.bias_ih",   "decoder.rnn.bias_hh",
    "decoder.decoder.2.weight", "decoder.decoder.2.bias",
}


def find_then_branch(model):
    """Locate the subgraph that fires when sr == 16000."""
    for node in model.graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH and attr.name == "then_branch":
                    return attr.g
    raise RuntimeError("No top-level If/then_branch found")


def strip_prefix(raw: str) -> str:
    """'If_0_then_branch__Inline_0__encoder.0.reparam_conv.weight' -> 'encoder.0.reparam_conv.weight'."""
    marker = "__Inline_0__"
    if marker in raw:
        return raw.split(marker, 1)[1]
    return raw


def collect_constants(graph):
    """Return {logical_name: numpy.ndarray(float32)} for every float Constant
    we care about, across this graph and its subgraphs."""
    out = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(a.t)
                    if arr.dtype != np.float32:
                        continue
                    logical = strip_prefix(node.output[0])
                    if logical in WANTED:
                        out[logical] = arr
        # Recurse into any subgraphs
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                for k, v in collect_constants(a.g).items():
                    out.setdefault(k, v)
    return out


def export(onnx_path: str, bin_path: str, manifest_path: str) -> None:
    model = onnx.load(onnx_path)
    then_branch = find_then_branch(model)
    weights = collect_constants(then_branch)

    missing = WANTED - set(weights.keys())
    if missing:
        raise RuntimeError(f"Missing expected tensors: {sorted(missing)}")

    # Deterministic order: alphabetical by logical name.
    tensors = []
    blobs = []
    offset = 0
    for name in sorted(weights):
        arr = np.ascontiguousarray(weights[name].astype(np.float32, copy=False).flatten())
        tensors.append({
            "name": name,
            "shape": list(weights[name].shape),
            "offset": offset,
            "length": int(arr.size),
        })
        blobs.append(arr)
        offset += arr.size

    blob = np.concatenate(blobs).astype(np.float32)
    Path(bin_path).write_bytes(blob.tobytes(order="C"))
    Path(manifest_path).write_text(json.dumps({"tensors": tensors}, indent=2))

    # Self-check
    raw = Path(bin_path).read_bytes()
    assert len(raw) == blob.size * 4, f"size mismatch: {len(raw)} vs {blob.size * 4}"
    reread = np.frombuffer(raw, dtype=np.float32)
    assert np.allclose(reread, blob), "round-trip mismatch"

    print(f"Wrote {len(tensors)} tensors, {blob.size} floats "
          f"({len(raw) / 1024 / 1024:.2f} MB) to {bin_path}")
    print(f"Manifest: {manifest_path}")
    for t in tensors:
        print(f"  {t['shape']!s:20s}  {t['name']}")


if __name__ == "__main__":
    onnx_in = sys.argv[1] if len(sys.argv) > 1 else "weights/silero_vad_v5.onnx"
    bin_out = onnx_in.replace(".onnx", ".bin")
    manifest_out = onnx_in.replace(".onnx", ".manifest.json")
    export(onnx_in, bin_out, manifest_out)
