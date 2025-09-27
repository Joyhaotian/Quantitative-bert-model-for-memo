import onnx
from onnx import helper, TensorProto
IN = "ernie3_nano.onnx"             
OUT = "ernie3_nano_fixed.onnx"
model = onnx.load(IN)
graph = model.graph
def add_cast_to_int32_before(node, input_idx: int):
    src = node.input[input_idx]
    cast_out = src + "_i32"
    cast = helper.make_node(
        "Cast", inputs=[src], outputs=[cast_out],
        to=TensorProto.INT32, name=src+"_to_i32"
    )
    node.input[input_idx] = cast_out
    return cast
new_nodes = []
for n in graph.node:
    if n.op_type in ("Gather", "GatherElements", "GatherND"):
        cast_node = add_cast_to_int32_before(n, 1) 
        new_nodes.append(cast_node)
    new_nodes.append(n)
graph.ClearField("node")
graph.node.extend(new_nodes)
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("Saved:", OUT)
