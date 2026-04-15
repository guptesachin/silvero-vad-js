"""Print SileroVAD ONNX model structure, recursing into subgraphs.

v5 stores everything inside a nested If(Equal(sr, 16000), then_branch, else_branch),
with weights as Constant nodes inside each branch. So we must walk subgraphs.
"""
import sys
import onnx
from onnx import numpy_helper


def _fmt_attrs(node):
    parts = []
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            parts.append(f"{a.name}={list(a.ints)}")
        elif a.type == onnx.AttributeProto.INT:
            parts.append(f"{a.name}={a.i}")
        elif a.type == onnx.AttributeProto.FLOAT:
            parts.append(f"{a.name}={a.f}")
        elif a.type == onnx.AttributeProto.STRING:
            parts.append(f"{a.name}={a.s.decode(errors='replace')}")
        elif a.type == onnx.AttributeProto.TENSOR:
            arr = numpy_helper.to_array(a.t)
            if arr.size <= 8:
                parts.append(f"{a.name}={arr.tolist()}")
            else:
                parts.append(f"{a.name}=<tensor shape={list(arr.shape)} dtype={arr.dtype}>")
    return f" [{', '.join(parts)}]" if parts else ""


def walk(graph, prefix="", depth=0):
    indent = "  " * depth
    print(f"\n{indent}--- GRAPH {prefix or '(root)'} ---")
    for i, node in enumerate(graph.node):
        name = node.name or ""
        print(f"{indent}[{i:3d}] {node.op_type:15s} name={name} "
              f"in=[{', '.join(node.input)}] -> out=[{', '.join(node.output)}]{_fmt_attrs(node)}")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                walk(attr.g, prefix=f"{prefix}/{node.op_type}.{attr.name}", depth=depth + 1)


def summarize_constants(graph, prefix="", accum=None):
    """Collect {fully-qualified name: (shape, dtype, sample_value)}."""
    if accum is None:
        accum = {}
    for node in graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(a.t)
                    key = f"{prefix}/{node.output[0]}"
                    accum[key] = (list(arr.shape), str(arr.dtype), arr.size)
        for a in node.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                summarize_constants(a.g, f"{prefix}/{node.op_type}.{a.name}", accum)
    return accum


def main(path: str) -> None:
    model = onnx.load(path)
    print("=" * 70)
    print("GRAPH INPUTS / OUTPUTS")
    print("=" * 70)
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  IN  {inp.name}: {dims}")
    for out in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  OUT {out.name}: {dims}")

    print("\n" + "=" * 70)
    print("CONSTANT TENSOR INVENTORY (recursive)")
    print("=" * 70)
    consts = summarize_constants(model.graph)
    # Show only the meaningful weight-ish constants (size > 1)
    weight_like = {k: v for k, v in consts.items() if v[2] > 1}
    for k, (shape, dtype, size) in sorted(weight_like.items()):
        print(f"  {shape!s:25s} {dtype:10s} size={size:8d}  {k}")
    total = sum(v[2] for v in weight_like.values())
    print(f"\n  TOTAL weight floats: {total} (~{total * 4 / 1024 / 1024:.2f} MB)")

    print("\n" + "=" * 70)
    print("FULL GRAPH WALK")
    print("=" * 70)
    walk(model.graph)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights/silero_vad_v5.onnx")
