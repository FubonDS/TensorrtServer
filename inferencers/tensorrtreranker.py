import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from transformers import AutoTokenizer


class TensorRTReranker:
    def __init__(self, engine_path: str, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

        input_shape = self.engine.get_tensor_shape(self.input_names[0])
        if -1 in tuple(input_shape):
            print("[INFO] Engine uses dynamic shapes, need to set_input_shape later.")
            self.dynamic = True
            for name in self.input_names:
                profile_idx = 0
                min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, profile_idx)
                self.max_batch_size = max_shape[0]
                self.max_length = max_shape[1]
                print(f"[INFO] Input '{name}' dynamic range: min={min_shape}, opt={opt_shape}, max={max_shape}")
        else:
            self.dynamic = False
            self.max_batch_size, self.max_length = tuple(input_shape)
            print(f"[INFO] Engine config: batch={self.max_batch_size}, seq_len={self.max_length}")

        self.bindings = {}
        if not self.dynamic:
            for name in self.input_names + self.output_names:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_mem = cuda.pagelocked_empty(shape, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[name] = (host_mem, device_mem)

        self.stream = cuda.Stream()
        print(f"[INFO] Reranker model loaded. Dynamic: {self.dynamic}")
    
    def infer(self, pairs: list[tuple[str, str]]):
        orig_n = len(pairs)
        if self.dynamic and orig_n > self.max_batch_size:
            print(f"[INFO] Batch size {orig_n} exceeds max_batch_size {self.max_batch_size}, splitting...")
            scores, elapsed_ms = [], 0
            for i in range(0, orig_n, self.max_batch_size):
                sub_pairs = pairs[i:i + self.max_batch_size]
                sub_scores, sub_elapsed = self.infer(sub_pairs)
                scores.extend(sub_scores)
                elapsed_ms += sub_elapsed
            return scores, elapsed_ms
        
        if self.dynamic:
            batch_size = len(pairs)
            max_length = self.max_length
            self.context.set_input_shape("input_ids", (batch_size, max_length))
            self.context.set_input_shape("attention_mask", (batch_size, max_length))

            self.bindings = {}
            for name in self.input_names + self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                host_mem = cuda.pagelocked_empty(shape, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings[name] = (host_mem, device_mem)
        else:
            batch_size = self.max_batch_size
            max_length = self.max_length
            assert len(pairs) <= batch_size
            pad_n = batch_size - len(pairs)
            if pad_n > 0:
                pairs = pairs + [("", "")] * pad_n

        enc = self.tokenizer(
            pairs,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)

        np.copyto(self.bindings["input_ids"][0], input_ids)
        np.copyto(self.bindings["attention_mask"][0], attention_mask)

        for name in self.input_names:
            cuda.memcpy_htod_async(self.bindings[name][1], self.bindings[name][0], self.stream)

        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, int(self.bindings[name][1]))

        start = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[name][0], self.bindings[name][1], self.stream)

        self.stream.synchronize()
        elapsed_ms = (time.time() - start) * 1000

        scores = self.bindings[self.output_names[0]][0][:orig_n].tolist()
        print(f"[DEBUG] Inference completed in {elapsed_ms:.2f} ms for batch size {orig_n}")
        return scores, elapsed_ms


if __name__ == "__main__":
    model = TensorRTReranker(
        engine_path="bge_reranker_large_dynamic_bs.trt",
        tokenizer_path="bge-reranker-large-tokenizer"
    )
    
    pairs = [
        ("Theory is essential for understanding machine learning.", "Machine learning is taught best through projects."),
        ("Theory is essential for understanding machine learning.", "A fast, dark-colored fox leaps over a sleepy canine.")
    ]
    
    scores, elapsed = model.infer(pairs)
    print("Scores:", scores)
    print(f"Scores len: {len(scores)}")
    print(f"Elapsed: {elapsed:.2f} ms")
    
    # batch = 35
    # query = "Theory is essential for understanding machine learning."
    # doc = "Machine learning is taught best through projects."
    # docs = [doc] * batch

    # scores, elapsed = model.infer(query, docs)
    # print("Scores:", scores)
    # print(f"Scores len: {len(scores)}")
    # print(f"Elapsed: {elapsed:.2f} ms")
