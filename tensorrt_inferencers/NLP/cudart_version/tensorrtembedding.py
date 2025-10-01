import time

import numpy as np
from cuda.bindings import runtime as cudart
from transformers import AutoTokenizer

from .baseinferencer import BaseTensorrtInferencer


class TensorRTEmbedding(BaseTensorrtInferencer):
    """
    TensorRT Embedding 推理器

    Args:
        engine_path (str): TensorRT 引擎檔案路徑
        tokenizer_path (str): 分詞器模型名稱或路徑，預設為 "bge-m3-tokenizer"
    """
    
    def __init__(self, engine_path: str, tokenizer_path: str = "bge-m3-tokenizer", reuse_dynamic_buffer: bool = True):
        super().__init__(engine_path, reuse_dynamic_buffer=reuse_dynamic_buffer)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    
    def build_cuda_graph(self, batch_size: int, max_length: int = None):
        if max_length is None:
            max_length = self.max_length

        if self.dynamic:
            self.context.set_input_shape("input_ids", (batch_size, max_length))
            self.context.set_input_shape("attention_mask", (batch_size, max_length))

        # dummy data
        doc = "graph warmup document"
        documents = [doc] * batch_size

        enc = self.tokenizer(documents,
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


    def infer(self, documents: list[str], cuda_graph=True):
        if cuda_graph and not self.graphs:
            raise RuntimeError("[ERROR] No CUDA graph found, please run build_cuda_graph first.")
        
        orig_n = len(documents)

        if self.dynamic and orig_n > self.max_batch_size:
            self.logger.info(f"[INFO] Batch size {orig_n} exceeds max_batch_size {self.max_batch_size}, splitting...")
            embeddings, elapsed_ms = [], 0
            for i in range(0, orig_n, self.max_batch_size):
                sub_documents = documents[i:i + self.max_batch_size]
                sub_embeddings, sub_elapsed = self.infer(sub_documents)
                embeddings.extend(sub_embeddings)
                elapsed_ms += sub_elapsed
            return embeddings, elapsed_ms

        if self.dynamic:
            max_length = self.max_length
            batch_size = len(documents)
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
            assert len(documents) <= batch_size

            pad_n = batch_size - len(documents)
            if pad_n > 0:
                documents = documents + [""] * pad_n

        enc = self.tokenizer(
            documents,
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
        
        embeddings = self.buffers_host[self.output_names[0]][:orig_n].tolist()
        return embeddings, elapsed_ms