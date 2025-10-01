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
def test_batch_size_no_cuda_graph(batch_size):
    """測試特定 batch_size 的非 CUDA Graph 性能"""
    print(f"\n=== 測試 Batch Size: {batch_size} (No CUDA Graph) ===")
    
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
        buffers_host[name] = host_buf

        err, dev_ptr = cudart.cudaMallocAsync(host_buf.nbytes, stream)
        buffers_device[name] = dev_ptr

        context.set_tensor_address(name, dev_ptr)

    np.copyto(buffers_host["input_ids"], enc["input_ids"])
    np.copyto(buffers_host["attention_mask"], enc["attention_mask"])

    output_name = [n for n in buffers_host if n not in ["input_ids", "attention_mask"]][0]

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

print("開始測試不同 Batch Size 的非 CUDA Graph 性能...")
print("=" * 60)

for bs in batch_sizes:
    try:
        avg_time, throughput = test_batch_size_no_cuda_graph(bs)
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
print("測試結果總結 (No CUDA Graph):")
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
============================================================
測試結果總結 (No CUDA Graph):
============================================================
Batch Size   總時間(ms)      吞吐量(samples/s)     單樣本時間(ms)      
------------------------------------------------------------
1            4.83         206.84             4.83           
2            5.28         378.69             2.64           
4            6.08         657.97             1.52           
8            8.74         915.86             1.09           
16           16.27        983.13             1.02           
32           31.07        1030.09            0.97 
"""