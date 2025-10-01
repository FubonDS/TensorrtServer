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
def test_batch_size(batch_size):
    """測試特定 batch_size 的 CUDA Graph 性能"""
    print(f"\n=== 測試 Batch Size: {batch_size} ===")
    
    premises = ["A man inspects the uniform of a figure in some East Asian country."] * batch_size
    hypotheses = ["The man is sleeping."] * batch_size

    max_length = 256
    enc = tokenizer(
        premises, hypotheses,
        return_tensors="np",
        padding="max_length", truncation=True, max_length=max_length
    )

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
        err, dev_ptr = cudart.cudaMallocAsync(host_buf.nbytes, stream)
        assert err == 0, f"cudaMalloc failed for {name}"

        buffers_host[name] = host_buf
        buffers_device[name] = dev_ptr
        context.set_tensor_address(name, dev_ptr)

    np.copyto(buffers_host["input_ids"], enc["input_ids"])
    np.copyto(buffers_host["attention_mask"], enc["attention_mask"])

    for name in ["input_ids", "attention_mask"]:
        err = cudart.cudaMemcpyAsync(
            buffers_device[name],                 
            buffers_host[name].ctypes.data,       
            buffers_host[name].nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )

    err = cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
    context.execute_async_v3(stream_handle=stream)
    err, graph = cudart.cudaStreamEndCapture(stream)
    assert err == 0, f"StreamEndCapture failed {err}"

    err, graph_exec = cudart.cudaGraphInstantiate(graph, 0)
    assert err == 0, f"GraphInstantiate failed {err}"

    for _ in range(5):  
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)

    N = 20
    t0 = time.time()
    for _ in range(N):
        cudart.cudaGraphLaunch(graph_exec, stream)
        cudart.cudaStreamSynchronize(stream)
    t1 = time.time()

    avg_time = (t1 - t0) * 1000 / N
    throughput = batch_size / (avg_time / 1000)
    
    print(f"平均推理時間: {avg_time:.2f} ms")
    print(f"每秒處理樣本數: {throughput:.2f} samples/sec")
    print(f"每樣本平均時間: {avg_time / batch_size:.2f} ms/sample")

    for ptr in buffers_device.values():
        cudart.cudaFree(ptr)
    cudart.cudaStreamDestroy(stream)
    
    return avg_time, throughput

batch_sizes = [1, 2, 4, 8, 16, 32]
results = []

print("開始測試不同 Batch Size 的 CUDA Graph 性能...")
print("=" * 60)

for bs in batch_sizes:
    try:
        avg_time, throughput = test_batch_size(bs)
        results.append({
            'batch_size': bs,
            'avg_time_ms': avg_time,
            'throughput_samples_per_sec': throughput,
            'time_per_sample_ms': avg_time / bs
        })
    except Exception as e:
        print(f"Batch size {bs} 測試失敗: {e}")
        continue

print("\n" + "=" * 60)
print("測試結果總結:")
print("=" * 60)
print(f"{'Batch Size':<12} {'總時間(ms)':<12} {'吞吐量(samples/s)':<18} {'單樣本時間(ms)':<15}")
print("-" * 60)

for result in results:
    print(f"{result['batch_size']:<12} {result['avg_time_ms']:<12.2f} {result['throughput_samples_per_sec']:<18.2f} {result['time_per_sample_ms']:<15.2f}")

if results:
    best_throughput = max(results, key=lambda x: x['throughput_samples_per_sec'])
    best_latency = min(results, key=lambda x: x['time_per_sample_ms'])
    
    print(f"\n最佳吞吐量: Batch Size {best_throughput['batch_size']} ({best_throughput['throughput_samples_per_sec']:.2f} samples/sec)")
    print(f"最佳延遲: Batch Size {best_latency['batch_size']} ({best_latency['time_per_sample_ms']:.2f} ms/sample)")
"""
CUDA Graph)
Batch Size   總時間(ms)      吞吐量(samples/s)     單樣本時間(ms)      
------------------------------------------------------------
1            4.24         235.68             4.24           
2            4.70         425.08             2.35           
4            5.53         723.68             1.38           
8            8.38         954.31             1.05           
16           15.68        1020.65            0.98           
32           30.47        1050.38            0.95    
"""