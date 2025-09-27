📘 Quantitative-BERT Model for Memo

This repository provides scripts and converted models for using ERNIE-3.0 Nano in memo classification tasks.
It includes both ONNX (PC client) and TFLite (mobile client) versions with fully built-in BERT support.

✨ Overview

Base model: nghuyong/ernie-3.0-nano-zh

Conversion pipeline:

Hugging Face PyTorch → ONNX (PC)

ONNX → TensorFlow SavedModel

TensorFlow → TFLite (mobile)

The converted models are used to classify user-written memos directly on client devices.

model_quint8_avx2.onnx: Quantized ONNX model optimized for desktop clients.

ernie3_nano_select.tflite: Quantized TFLite model optimized for mobile clients.

⚙️ Environment Setup

We recommend Python 3.10 on Linux/CPU for reproducibility.

python -m pip install -U pip setuptools wheel
python -m pip install "transformers==4.41.1" "torch>=2.1,<3"
python -m pip install "onnx==1.14.0" "onnxruntime==1.15.1" "onnx-simplifier==0.4.33" "onnx-tf==1.10.0"
python -m pip install "tensorflow==2.13.0" "tensorflow-probability==0.21.0"
# Fix typing-extensions if TensorFlow downgrades it
python -m pip install --no-deps "typing-extensions==4.12.2"

🛠 Conversion Pipeline
1. Export ONNX (with approximate GELU)

Script: export_onnx.py

Uses gelu_new to avoid Erf.

Fixes sequence length and sets opset_version=12.

2. Fix Gather indices → int32

Script: onnx_fix_indices.py

Ensures Gather/GatherND use int32 indices.

3. Downgrade Unsqueeze ops

Script: downgrade_unsqueeze_to11.py

Converts Unsqueeze-13 → Unsqueeze-11.

Sets global opset to 12.

4. Convert ONNX → TensorFlow

Script: onnx_to_tf.py

Converts ONNX to TensorFlow SavedModel using onnx-tf.

5. Convert TensorFlow → TFLite

Script: tf_to_tflite.py

Produces .tflite with only TFLITE_BUILTINS ops.

Optional optimizations: float16 / default quantization.

📂 Repository Structure
Quantitative-bert-model-for-memo/
│
├── model_quint8_avx2.onnx          # Quantized ONNX model (desktop)
├── ernie3_nano_select.tflite       # Quantized TFLite model (mobile)
├── sentencepiece.bpe.model         # Tokenizer model
├── tokenizer.json                  # Hugging Face tokenizer config
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
│
├── export_onnx.py                  # Step 1: PyTorch → ONNX
├── onnx_fix_indices.py             # Step 2: Fix indices
├── downgrade_unsqueeze_to11.py     # Step 3: Downgrade ops
├── onnx_to_tf.py                   # Step 4: ONNX → TensorFlow
├── tf_to_tflite.py                 # Step 5: TF → TFLite
│
└── LICENSE

📌 Key Notes

No Erf ops → safer for mobile deployment.

Only TFLITE_BUILTINS ops used (avoids Select TF Ops).

Fixed-length export ensures deterministic behavior.

Optimized for classification of short text memos on device.

📜 License

Apache 2.0 License. See LICENSE
 for details.
