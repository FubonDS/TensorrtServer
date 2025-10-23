import time
from typing import List, Tuple, Any      
import numpy as np
from cuda.bindings import runtime as cudart
from transformers import AutoTokenizer

from .baseinferencer import BaseTensorrtInferencer


class TensorRTNLI(BaseTensorrtInferencer):
    def __init__(self, engine_path: str, tokenizer_path: str = "joeddav/xlm-roberta-large-xnli", reuse_dynamic_buffer: bool = True):
        super().__init__(engine_path, reuse_dynamic_buffer=reuse_dynamic_buffer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        self.labels = ["contradiction", "neutral", "entailment"]

    def build_cuda_graph(self, batch_size: int, max_length: int = None):
        if max_length is None:
            max_length = self.max_length

        if self.dynamic:
            self.context.set_input_shape("input_ids", (batch_size, max_length))
            self.context.set_input_shape("attention_mask", (batch_size, max_length))

        # dummy data
        dummy_premises = ["graph warmup premise"] * batch_size
        dummy_hypotheses = ["graph warmup hypothesis"] * batch_size

        enc = self.tokenizer(dummy_premises, dummy_hypotheses,
                             return_tensors="np",
                             padding="max_length", truncation=True, max_length=max_length)
        np.copyto(self.buffers_host["input_ids"][:batch_size], enc["input_ids"])
        np.copyto(self.buffers_host["attention_mask"][:batch_size], enc["attention_mask"])

        for name in self.input_names:
            err, = cudart.cudaMemcpyAsync(
                self.buffers_device[name],
                self.buffers_host[name].ctypes.data,
                self.buffers_host[name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            )

        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, int(self.buffers_device[name]))

        self.context.execute_async_v3(stream_handle=self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        # Capture
        cudart.cudaStreamBeginCapture(self.stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        self.context.execute_async_v3(stream_handle=self.stream)
        err, graph = cudart.cudaStreamEndCapture(self.stream)
        err, graph_exec = cudart.cudaGraphInstantiate(graph, 0)

        self.graphs[batch_size] = graph_exec
        self.logger.info(f"[INFO] CUDA Graph built for batch_size={batch_size}")

    def infer(self, premises, hypotheses, cuda_graph=True):
        if cuda_graph and not self.graphs:
            raise RuntimeError("[ERROR] No CUDA graph found, please run build_cuda_graph first.")
        
        assert len(premises) == len(hypotheses), "premises 與 hypotheses 長度必須一致"

        orig_n = len(premises)

        if self.dynamic and orig_n > self.max_batch_size:
            self.logger.info(f"[INFO] Batch size {orig_n} exceeds max_batch_size {self.max_batch_size}, splitting...")
            preds, logits, elapsed_ms = [], [], 0
            for i in range(0, orig_n, self.max_batch_size):
                sub_premises = premises[i:i + self.max_batch_size]
                sub_hypotheses = hypotheses[i:i + self.max_batch_size]
                sub_preds, sub_logits, sub_elapsed = self.infer(sub_premises, sub_hypotheses)
                preds.extend(sub_preds)
                logits.append(sub_logits)
                elapsed_ms += sub_elapsed
            logits = np.vstack(logits)  
            return preds, logits, elapsed_ms

        if self.dynamic:
            max_length = self.max_length
            batch_size = len(premises)
            self.context.set_input_shape("input_ids", (batch_size, max_length))
            self.context.set_input_shape("attention_mask", (batch_size, max_length))
            if not self.reuse_dynamic_buffer:
                self.allocate_buffers({
                    "input_ids": (batch_size, max_length),
                    "attention_mask": (batch_size, max_length),
                })

        else:
            batch_size = self.max_batch_size
            max_length = self.max_length
            assert len(premises) <= batch_size

            pad_n = batch_size - len(premises)
            if pad_n > 0:
                premises = premises + ["dummy premise"] * pad_n
                hypotheses = hypotheses + ["dummy hypothesis"] * pad_n

        enc = self.tokenizer(
            premises, hypotheses,
            return_tensors="np",
            padding="max_length", truncation=True, max_length=max_length
        )

        input_ids = self._cast_to_engine_dtype("input_ids", enc["input_ids"])
        attention_mask = self._cast_to_engine_dtype("attention_mask", enc["attention_mask"])

        np.copyto(self.buffers_host["input_ids"][:batch_size], input_ids)
        np.copyto(self.buffers_host["attention_mask"][:batch_size], attention_mask)

        for name in self.input_names:
            err, = cudart.cudaMemcpyAsync(
                self.buffers_device[name],
                self.buffers_host[name].ctypes.data,
                self.buffers_host[name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            )

        for name in self.input_names + self.output_names:
            self.context.set_tensor_address(name, int(self.buffers_device[name]))

        start = time.time()
        if batch_size in self.graphs and cuda_graph:
            self.logger.info(f"[INFO] Using CUDA Graph for batch_size={batch_size}")
            cudart.cudaGraphLaunch(self.graphs[batch_size], self.stream)
        else:
            self.context.execute_async_v3(stream_handle=self.stream)

        for name in self.output_names:
            err, = cudart.cudaMemcpyAsync(
                self.buffers_host[name].ctypes.data,
                self.buffers_device[name],
                self.buffers_host[name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )

        cudart.cudaStreamSynchronize(self.stream)
        elapsed_ms = (time.time() - start) * 1000

        logits = self.buffers_host[self.output_names[0]][:orig_n].reshape(orig_n, -1)

        preds = [self.labels[np.argmax(l)] for l in logits[:orig_n]]

        return preds, logits[:orig_n], elapsed_ms