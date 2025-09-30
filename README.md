📘 Quantitative-BERT Model for Memo

Scripts and converted models to run a compact ERNIE / BERT-style keyword-classification model on desktop and mobile. Includes reproducible conversion pipeline: PyTorch → ONNX → TensorFlow SavedModel → TFLite (post-training quantization).
It includes both ONNX (PC client) and TFLite (mobile client) versions with fully built-in BERT support.
MeMo (note app) — published on Microsoft Store and Xiaomi App Store.  
- Microsoft Store: https://apps.microsoft.com/detail/9N8BJ58F1Q24  
- Xiaomi App Store: https://app.mi.com/details?id=com.tiantian.memo&ref=search

✨ Overview

 Highlights / Features
- Compact on-device models: Quantized TFLite for mobile (reduced model size for CPU inference).  
- Reproducible conversion pipeline: scripts to export PyTorch → ONNX → TF SavedModel → TFLite.  
- Mobile-ready operator: TFLITE_BUILTINS only (avoids custom TF ops).  
- Useful utilities: tokenizer files, example quantized ONNX for desktop (`model_quint8_avx2.onnx`).  
- Integration examples: guidance to embed TFLite models into Android/Flutter clients.

 Quick start (run on Linux / CPU)
 > We recommend Python 3.10 on Linux/CPU for reproducibility.

1. Clone & LFS
 git clone https://github.com/Joyhaotian/Quantitative-bert-model-for-memo.git
 git lfs pull   # necessary to download large model files tracked by LFS

2.Create venv and install deps
 python -m venv .venv
 source .venv/bin/activate
 python -m pip install -U pip setuptools wheel
 pip install transformers==4.41.1 torch>=2.1,<3 onnx==1.14.0 onnxruntime==1.15.1 onnx-simplifier onnx-tf==1.10.0 tensorflow==2.13.0
 pip install --no-deps "typing-extensions==4.12.2"
 
3.Export ONNX from a HuggingFace checkpoint
 python export_onnx.py \
  --model nghuyong/ernie-3.0-nano-zh \
  --output model_quint8_avx2.onnx \
  --seq_len 128
  
4.Optional: ONNX → TF → TFLite
  python onnx_fix_indices.py --input model_quint8_avx2.onnx --output model_fixed.onnx
  python onnx_to_tf.py --input model_fixed.onnx --output saved_model_dir
  python tf_to_tflite.py --saved_model_dir saved_model_dir --output ernie3_nano_select.tflite --quantize
