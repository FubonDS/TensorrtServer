import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from .baseinferencer import BaseTensorrtInferencer


class TensorRTReranker(BaseTensorrtInferencer):
    def infer(self, pairs: list[tuple[str, str]]):
        orig_n = len(pairs)
        if self.dynamic and orig_n > self.max_batch_size:
            self.logger.info(f"[INFO] Batch size {orig_n} exceeds max_batch_size {self.max_batch_size}, splitting...")
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
        return scores, elapsed_ms