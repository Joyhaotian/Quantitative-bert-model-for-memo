import onnx
from onnx import helper, numpy_helper
IN = "ernie3_nano_fixed.onnx"
OUT = "ernie3_nano_op12.onnx"
model = onnx.load(IN)
graph = model.graph
name_to_init = {init.name: init for init in graph.initializer}
const_value_map = {}
for n in graph.node:
    if n.op_type == "Constant":
        for a in n.attribute:
            if a.name == "value":
                const_value_map[n.output[0]] = numpy_helper.to_array(a.t)
def get_const_axes(name):
    if name in name_to_init:
        return numpy_helper.to_array(name_to_init[name])
    if name in const_value_map:
        return const_value_map[name]
    return None
new_nodes, changed = [], 0
for n in graph.node:
    if n.op_type == "Unsqueeze" and len(n.input) == 2:
        axes_arr = get_const_axes(n.input[1])
        if axes_arr is None:
            new_nodes.append(n)
            continue
        axes = [int(x) for x in axes_arr.flatten().tolist()]
        new_n = helper.make_node("Unsqueeze", inputs=[n.input[0]],
                                 outputs=list(n.output), name=n.name or "")
        new_n.attribute.extend([helper.make_attribute("axes", axes)])
        new_nodes.append(new_n); changed += 1
    else:
        new_nodes.append(n)
graph.ClearField("node")
graph.node.extend(new_nodes)
found = False
for imp in model.opset_import:
    if (imp.domain or "") == "":
        imp.version = 12
        found = True
if not found:
    model.opset_import.extend([helper.make_operatorsetid("", 12)])
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(f"done. downgraded Unsqueeze nodes: {changed}; saved -> {OUT}")
