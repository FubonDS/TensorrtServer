from transformers import AutoModel, AutoTokenizer
import torch
import onnx
import numpy as np

def convert_embedding_to_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = "/data/data_science_department/model/embedding_engine/model/embedding_model/bge-m3-model"  
    tokenizer_path = "/data/data_science_department/model/embedding_engine/model/embedding_model/bge-m3-tokenizer"
    
    embedding_model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    embedding_model.eval()
    
    sample_text = "自然語言處理是人工智慧和計算機科學領域的分支，專注於使計算機能夠理解、解釋和生成人類語言。"
    encoded_input = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(device)
    
    dummy_input_ids = encoded_input['input_ids']
    dummy_attention_mask = encoded_input['attention_mask']
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["embeddings"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "embeddings": {0: "batch_size"}
    }
    
    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings
    
    wrapped_model = EmbeddingWrapper(embedding_model)
    
    onnx_path = "./embedding_models/model_dynamic/bge_m3_embedding_dynamic.onnx"
    
    print("Converting model to ONNX...")
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids.to(torch.int32), dummy_attention_mask.to(torch.int32)),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        use_external_data_format=False,
    )
    
    print(f"Model exported to {onnx_path}")
    
    return onnx_path, tokenizer

if __name__ == "__main__":
    onnx_path, tokenizer = convert_embedding_to_onnx()