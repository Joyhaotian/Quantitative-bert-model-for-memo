import onnx
from onnx_tf.backend import prepare
IN = "ernie3_nano_op12.onnx"
OUT_DIR = "ernie3_nano_tf"
onnx_model = onnx.load(IN)
tf_rep = prepare(onnx_model, strict=False)  
tf_rep.export_graph(OUT_DIR)
print("SavedModel:", OUT_DIR)
