"""Reject incompatible ONNX operations even when Python assertions are disabled."""
from pathlib import Path
import tempfile

import onnx

import graph as G
import gen_variants as V


def main():
    source = (V.KERNELS / V.SOURCE).read_text().replace(V.LAUNCH, "// anchor removed")
    try:
        V.generate("relu", source)
    except ValueError as failure:
        if "anchor moved" not in str(failure):
            raise
    else:
        raise AssertionError("generator accepted a missing launch anchor")
    with tempfile.TemporaryDirectory(prefix="scrfd-graph-test-") as directory:
        path = Path(directory) / "unsupported.onnx"
        for attribute, value in (("group", 2), ("dilations", [1, 2]), ("strides", [1, 2]),
                                 ("pads", [1, 0, 1, 0])):
            model = onnx.load(str(G.model_path()))
            conv = next(node for node in model.graph.node if node.op_type == "Conv")
            kept = [a for a in conv.attribute if a.name != attribute]
            del conv.attribute[:]
            conv.attribute.extend(kept + [onnx.helper.make_attribute(attribute, value)])
            onnx.save(model, str(path))
            try:
                G.load(path=path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"accepted incompatible convolution {attribute}")
    print("  PASS graph and generator checks reject unsupported inputs under python -O")


if __name__ == "__main__":
    main()
