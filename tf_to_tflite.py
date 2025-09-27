import tensorflow as tf
SAVED = "ernie3_nano_tf"
OUT = "ernie3_nano.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# 体积优化（可选）
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
tflite = converter.convert()
open(OUT, "wb").write(tflite)
print("OK ->", OUT)
