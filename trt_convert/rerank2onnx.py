from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
import torch
import numpy as np

def convert_reranker_to_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = "/home/max/vllm-server/llm-router-server/embedding_reranker_server/embedding_engine/model/reranking_model/bge-reranker-large-model"
    tokenizer_path = "/home/max/vllm-server/llm-router-server/embedding_reranker_server/embedding_engine/model/reranking_model/bge-reranker-large-tokenizer"

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    query = "Theory is essential for understanding machine learning."
    documents = [
                "Machine learning is taught best through projects.",
            ]
    model.eval()
    
    encoded_input = tokenizer(
        [[query, documents[0]]],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(device)
    
    dummy_input_ids = encoded_input['input_ids']
    dummy_attention_mask = encoded_input['attention_mask']
    
    input_names = ["input_ids", "attention_mask"]
    output_names = ["scores"]
    
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "scores": {0: "batch_size"}
    }
    
    class RerankerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits.squeeze(-1)
            return scores

    wrapped_model = RerankerWrapper(model)

    onnx_path = "./model_dynamic/bge_reranker_large_dynamic.onnx"
    
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
    onnx_path, tokenizer = convert_reranker_to_onnx()