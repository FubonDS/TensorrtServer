import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


class BaseTensorrtInferencer(ABC):
    def __init__(
        self, 
        engine_path: str, 
        log_level: int = logging.INFO,
        reuse_dynamic_buffer: bool = True, 
    ):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        self.reuse_dynamic_buffer = reuse_dynamic_buffer

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"[ERROR] Engine file not found: {engine_path}")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                             if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

        # build bindings and allocate buffers
        self.buffers_host = {}
        self.buffers_device = {}

        err, self.stream = cudart.cudaStreamCreate()
        
        input_shape = self.engine.get_tensor_shape(self.input_names[0])
        if -1 in tuple(input_shape): 
            self.logger.info("[INFO] Engine uses dynamic shapes, need to set_input_shape later.")
            self.dynamic = True
            for name in self.input_names:
                profile_idx = 0  
                min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, profile_idx)
                self.max_batch_size = max_shape[0]
                self.max_length = max_shape[1]
                self.logger.info(f"[INFO] Input '{name}' dynamic range: min={min_shape}, opt={opt_shape}, max={max_shape}")

            if self.reuse_dynamic_buffer:
                self.logger.info(f"[INFO] Pre-allocating max buffer: batch={self.max_batch_size}, seq_len={self.max_length}")
                self.allocate_buffers({
                    "input_ids": (self.max_batch_size, self.max_length),
                    "attention_mask": (self.max_batch_size, self.max_length),
                })
        else:
            self.dynamic = False
            self.max_batch_size, self.max_length = tuple(input_shape)
            self.logger.info(f"[INFO] Engine config: batch={self.max_batch_size}, seq_len={self.max_length}")
            self._allocate_static_buffers()
                
        self.logger.info(f"[INFO] Model loaded. Dynamic: {self.dynamic}")
        
    def _allocate_static_buffers(self):
        for name in self.input_names + self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buffer = np.zeros(shape, dtype=dtype)
            self.buffers_host[name] = host_buffer

            err, dev_ptr = cudart.cudaMallocAsync(host_buffer.nbytes, self.stream)
            self.buffers_device[name] = dev_ptr

    def allocate_buffers(self, shape_dict: dict[str, tuple[int, ...]]):
        
        for name in self.input_names + self.output_names:
            if name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                
            if -1 in shape:
                shape = tuple(self.max_batch_size if d == -1 else d for d in shape)
                
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buffer = np.zeros(shape, dtype=dtype)
            self.buffers_host[name] = host_buffer

            err, dev_ptr = cudart.cudaMallocAsync(host_buffer.nbytes, self.stream)
            self.buffers_device[name] = dev_ptr
            
    def _cast_to_engine_dtype(self, name: str, array: np.ndarray) -> np.ndarray:
        expected_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
        if array.dtype != expected_dtype:
            self.logger.debug(f"[DEBUG] Casting '{name}' from {array.dtype} -> {expected_dtype}")
            return array.astype(expected_dtype)
        return array
        
    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        pass

    def close(self):
        """Safely synchronize and free CUDA / TRT resources. Call explicitly on shutdown."""
        if getattr(self, "_closed", False):
            return
        try:
            if hasattr(self, "context"):
                try:
                    del self.context
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, "engine"):
                try:
                    del self.engine
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(self, "runtime"):
                try:
                    del self.runtime
                except Exception:
                    pass
        except Exception:
            pass

        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass