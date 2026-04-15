"""Compare numpy forward pass against ORT by extracting the 16kHz subgraph
as a standalone ONNX model and exposing intermediate tensors.
"""
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def load_weights():
    manifest = json.loads(Path("weights/silero_vad_v5.manifest.json").read_text())
    blob = np.frombuffer(Path("weights/silero_vad_v5.bin").read_bytes(), dtype=np.float32)
    w = {}
    for t in manifest["tensors"]:
        w[t["name"]] = blob[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"]).copy()
    return w


def extract_then_branch_as_model(src_onnx: str, extra_outputs: list[str], dst_onnx: str):
    """Promote the top-level If's then_branch subgraph into a standalone model.
    Adds extra_outputs (tensor names in the subgraph) as additional model outputs.
    """
    src = onnx.load(src_onnx)
    then_branch = None
    for node in src.graph.node:
        if node.op_type == "If":
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH and attr.name == "then_branch":
                    then_branch = attr.g
                    break
    assert then_branch is not None

    # The subgraph references `input` and `state` from the parent graph. Promote them
    # to subgraph inputs by adding value_infos with the right shapes.
    sub_inputs = [
        helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["N", "S"]),
        helper.make_tensor_value_info("state", onnx.TensorProto.FLOAT, [2, "N", 128]),
    ]
    # Rename subgraph outputs (they're currently unnamed or named "If_0_..._outputs_{0,1}");
    # we rename using .output field of the subgraph spec.
    sub_outputs = [
        helper.make_tensor_value_info(then_branch.output[0].name, onnx.TensorProto.FLOAT, None),
        helper.make_tensor_value_info(then_branch.output[1].name, onnx.TensorProto.FLOAT, None),
    ]
    for name in extra_outputs:
        sub_outputs.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))

    new_graph = helper.make_graph(
        nodes=list(then_branch.node),
        name="silero_16k",
        inputs=sub_inputs,
        outputs=sub_outputs,
        initializer=list(then_branch.initializer),
    )
    new_model = helper.make_model(new_graph, opset_imports=list(src.opset_import))
    new_model.ir_version = src.ir_version
    onnx.save(new_model, dst_onnx)


def list_tensor_names(onnx_path: str) -> list[str]:
    model = onnx.load(onnx_path)
    names = []
    for node in model.graph.node:
        names.extend(node.output)
    return names


def conv1d(x, W, b, stride, pad):
    C_in, L = x.shape
    C_out, _, K = W.shape
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad)))
    L_out = (x.shape[1] - K) // stride + 1
    out = np.zeros((C_out, L_out), dtype=np.float32)
    for oc in range(C_out):
        for t in range(L_out):
            out[oc, t] = float(b[oc]) + float(np.sum(W[oc] * x[:, t * stride:t * stride + K]))
    return out


def stft_mag(x_pad, basis, stride, n_freqs):
    K = basis.shape[2]
    L_out = (x_pad.size - K) // stride + 1
    mag = np.zeros((n_freqs, L_out), dtype=np.float32)
    for f in range(n_freqs):
        wr, wi = basis[f, 0, :], basis[n_freqs + f, 0, :]
        for t in range(L_out):
            seg = x_pad[t * stride:t * stride + K]
            re, im = float(np.dot(wr, seg)), float(np.dot(wi, seg))
            mag[f, t] = np.sqrt(re * re + im * im)
    return mag


def reflect_pad(x, p):
    left = x[1:p + 1][::-1]
    right = x[-p - 1:-1][::-1]
    return np.concatenate([left, x, right])


def main():
    w = load_weights()

    # Step 1: extract the 16kHz subgraph so we can expose intermediates.
    extract_then_branch_as_model("weights/silero_vad_v5.onnx", [], "/tmp/silero_16k.onnx")

    # Step 2: list all tensor names in the standalone model and find the ones we care about.
    all_names = list_tensor_names("/tmp/silero_16k.onnx")
    enc3_relu = next(n for n in all_names if "/encoder/3/activation/Relu" in n)
    stft_sqrt = next(n for n in all_names if "/stft/Sqrt_output_0" in n)
    print(f"Found enc3_relu: {enc3_relu}")
    print(f"Found stft_sqrt: {stft_sqrt}")

    # Step 3: patch the standalone model with extra outputs.
    extract_then_branch_as_model(
        "weights/silero_vad_v5.onnx",
        extra_outputs=[stft_sqrt, enc3_relu],
        dst_onnx="/tmp/silero_16k_patched.onnx",
    )

    sess = ort.InferenceSession("/tmp/silero_16k_patched.onnx", providers=["CPUExecutionProvider"])
    for i, o in enumerate(sess.get_outputs()):
        print(f"  out[{i}] name={o.name} shape={o.shape}")

    frame = np.zeros(512, dtype=np.float32)
    feeds = {
        "input": frame.reshape(1, -1),
        "state": np.zeros((2, 1, 128), dtype=np.float32),
    }
    outs = sess.run(None, feeds)
    prob_ort, state_ort, stft_ort, enc3_ort = outs[:4]
    print(f"\nORT prob: {float(prob_ort.flatten()[0]):.6f}")
    print(f"ORT stft shape: {stft_ort.shape}")
    print(f"ORT enc3 shape: {enc3_ort.shape}")

    # Numpy side
    padded = reflect_pad(frame, 32)
    mag_np = stft_mag(padded, w["stft.forward_basis_buffer"], stride=128, n_freqs=129)
    print(f"\nnp  stft shape: {mag_np.shape}")

    # stft_ort may be (N, 129, T). Compare.
    stft_ort_2d = stft_ort.squeeze(0) if stft_ort.ndim == 3 else stft_ort
    if stft_ort_2d.shape == mag_np.shape:
        print(f"STFT max abs diff: {np.abs(mag_np - stft_ort_2d).max():.6e}")
    else:
        print(f"STFT SHAPE MISMATCH: np={mag_np.shape} ort={stft_ort_2d.shape}")

    # Encoder output
    x = mag_np
    for (wn, bn, stride) in [
        ("encoder.0.reparam_conv.weight", "encoder.0.reparam_conv.bias", 1),
        ("encoder.1.reparam_conv.weight", "encoder.1.reparam_conv.bias", 2),
        ("encoder.2.reparam_conv.weight", "encoder.2.reparam_conv.bias", 2),
        ("encoder.3.reparam_conv.weight", "encoder.3.reparam_conv.bias", 1),
    ]:
        x = conv1d(x, w[wn], w[bn], stride=stride, pad=1)
        x = np.maximum(x, 0)
    enc_np = x
    enc_ort_2d = enc3_ort.squeeze(0) if enc3_ort.ndim == 3 else enc3_ort
    print(f"\nnp  enc3 shape: {enc_np.shape}")
    print(f"ort enc3 shape: {enc_ort_2d.shape}")
    if enc_np.shape == enc_ort_2d.shape:
        print(f"ENC3 max abs diff: {np.abs(enc_np - enc_ort_2d).max():.6e}")


if __name__ == "__main__":
    main()
