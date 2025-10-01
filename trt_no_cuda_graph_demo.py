import time
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart
from transformers import AutoTokenizer

logger = trt.Logger(trt.Logger.INFO)
with open("./nli_models/trt_models/nli_model_dynamic_bs.trt", "rb") as f:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
premises = ["A man inspects the uniform of a figure in some East Asian country."]
hypotheses = ["The man is sleeping."]

max_length = 256
enc = tokenizer(
    premises, hypotheses,
    return_tensors="np",
    padding="max_length", truncation=True, max_length=max_length
)

batch_size = 8
context.set_input_shape("input_ids", (batch_size, max_length))
context.set_input_shape("attention_mask", (batch_size, max_length))

buffers_host = {}
buffers_device = {}

err, stream = cudart.cudaStreamCreate()

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = tuple(context.get_tensor_shape(name))
    if -1 in shape:
        shape = tuple(batch_size if d == -1 else d for d in shape)

    host_buf = np.zeros(shape, dtype=dtype)
    buffers_host[name] = host_buf

    err, dev_ptr = cudart.cudaMallocAsync(host_buf.nbytes, stream)
    buffers_device[name] = dev_ptr

    context.set_tensor_address(name, dev_ptr)

np.copyto(buffers_host["input_ids"], enc["input_ids"])
np.copyto(buffers_host["attention_mask"], enc["attention_mask"])

for _ in range(5):
    for name in ["input_ids", "attention_mask"]:
        err, = cudart.cudaMemcpyAsync(
            buffers_device[name],
            buffers_host[name].ctypes.data,
            buffers_host[name].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )

    context.execute_async_v3(stream_handle=stream)

    output_name = [n for n in buffers_host if n not in ["input_ids", "attention_mask"]][0]
    err, = cudart.cudaMemcpyAsync(
        buffers_host[output_name].ctypes.data,
        buffers_device[output_name],
        buffers_host[output_name].nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )
    cudart.cudaStreamSynchronize(stream)

N = 20
t0 = time.time()
for _ in range(N):
    for name in ["input_ids", "attention_mask"]:
        err, = cudart.cudaMemcpyAsync(
            buffers_device[name],
            buffers_host[name].ctypes.data,
            buffers_host[name].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )

    context.execute_async_v3(stream_handle=stream)

    err, = cudart.cudaMemcpyAsync(
        buffers_host[output_name].ctypes.data,
        buffers_device[output_name],
        buffers_host[output_name].nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        stream
    )
    cudart.cudaStreamSynchronize(stream)
t1 = time.time()

print(f"[No CUDA Graph] batch=1 平均推理時間: {(t1 - t0) * 1000 / N:.2f} ms")

print("Logits shape:", buffers_host[output_name].shape)
print("First row:", buffers_host[output_name][0])

for ptr in buffers_device.values():
    cudart.cudaFree(ptr)
cudart.cudaStreamDestroy(stream)
