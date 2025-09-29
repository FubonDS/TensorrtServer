import logging
import os
from abc import ABC, abstractmethod
from typing import Any
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from transformers import AutoTokenizer


class BaseTensorrtInferencer(ABC):
    def __init__(
        self, 
        engine_path: str, 
        log_level: int = logging.INFO
    ):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(log_level)

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

        input_shape = self.engine.get_tensor_shape(self.input_names[0])
        if -1 in tuple(input_shape): 
            self.logger.info("[INFO] Engine uses dynamic shapes, need to set_input_shape later.")
            self.dynamic = True
            for name in self.input_names:
                profile_idx = 0  
                min_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[0]
                opt_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[1]
                max_shape = self.engine.get_tensor_profile_shape(name, profile_idx)[2]
                self.max_batch_size = max_shape[0]
                self.max_length = max_shape[1]
                self.logger.info(f"[INFO] Input '{name}' dynamic range: min={min_shape}, opt={opt_shape}, max={max_shape}")
        else:
            self.dynamic = False
            self.max_batch_size, self.max_length = tuple(input_shape)
            self.logger.info(f"[INFO] Engine config: batch={self.max_batch_size}, seq_len={self.max_length}")
            
        self.bindings = {}
        if not self.dynamic:
            self._allocate_static_buffers()
                
        self.stream = cuda.Stream()
        self.logger.info(f"[INFO] Model loaded. Dynamic: {self.dynamic}")
        
    def _allocate_static_buffers(self):
        for name in self.input_names + self.output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings[name] = (host_mem, device_mem)
            
    def allocate_buffers(self, shape_dict: dict[str, tuple[int, ...]]):
        self.bindings.clear()
        
        for name in self.input_names + self.output_names:
            if name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[name] = (host_mem, device_mem)
        
    @abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        pass