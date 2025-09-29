from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def convert_nli_to_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "joeddav/xlm-roberta-large-xnli"

    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.eval()
    
    dummy_inputs = tokenizer(
        ["Einstein was a scientist."] * 2,       
        ["Einstein was a chef."] * 2,             
        return_tensors="pt",
        padding="max_length",                     
        truncation=True,
        max_length=256                             
    ).to(device)
    
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"].to(torch.int32), dummy_inputs["attention_mask"].to(torch.int32)),
        "./model_dynamic_bs/nli_model_dynamic_bs.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        use_external_data_format=False,
        dynamic_axes={
            "input_ids": {0: "batch_size"},        # 設定 input_ids 的第 0 維為動態
            "attention_mask": {0: "batch_size"},   # 設定 attention_mask 的第 0 維為動態
            "logits": {0: "batch_size"}            # 設定 logits 的第 0 維為動態
        }
    )
    
    print(f"Model exported to {onnx_path}")
    
    return onnx_path, tokenizer

if __name__ == "__main__":
    onnx_path, tokenizer = convert_nli_to_onnx()