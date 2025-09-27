import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
MODEL_ID = "nghuyong/ernie-3.0-nano-zh"
SEQ_LEN = 128
ONNX_OUT = "ernie3_nano.onnx"
config = AutoConfig.from_pretrained(MODEL_ID)
config.hidden_act = "gelu_new"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, config=config).eval()
enc = tokenizer("你好", return_tensors="pt", padding="max_length",
                truncation=True, max_length=SEQ_LEN)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids, attention_mask, token_type_ids),
        ONNX_OUT,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        opset_version=12,              
        do_constant_folding=True,
        dynamic_axes=None,               )
print("Exported:", ONNX_OUT)