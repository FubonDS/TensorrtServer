## 1. 先將模型轉為 onnx
參考`trt_convert/embedding2onnx.py`

## 2.  在 tensorrt docker 進行轉換
- 靜態轉 trt
```bash
trtexec   --onnx=./nli_model/model_bs8/nli_model_bs8.onnx   --saveEngine=nli_model_bs8.trt   --fp16
```
- 動態轉 trt
```bash
trtexec \
  --onnx=./model_dynamic_bs/nli_model_dynamic_bs.onnx \
  --saveEngine=nli_model_dynamic_bs.trt \
  --fp16 \
  --minShapes=input_ids:1x256,attention_mask:1x256 \
  --optShapes=input_ids:8x256,attention_mask:8x256 \
  --maxShapes=input_ids:32x256,attention_mask:32x256
```