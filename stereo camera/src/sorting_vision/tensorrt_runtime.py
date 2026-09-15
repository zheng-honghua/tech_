from __future__ import annotations

from pathlib import Path

import numpy as np


class TensorRTCallable:
    """Small TensorRT 8.5+ / PyCUDA callable used only on the target Jetson."""

    def __init__(self, engine_path: str | Path) -> None:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt

        self.cuda = cuda
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.input_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        )
        self.stream = cuda.Stream()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        values = np.ascontiguousarray(batch, np.float32)
        self.context.set_input_shape(self.input_name, values.shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, np.float32)
        input_device = self.cuda.mem_alloc(values.nbytes)
        output_device = self.cuda.mem_alloc(output.nbytes)
        self.cuda.memcpy_htod_async(input_device, values, self.stream)
        self.context.set_tensor_address(self.input_name, int(input_device))
        self.context.set_tensor_address(self.output_name, int(output_device))
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT inference failed")
        self.cuda.memcpy_dtoh_async(output, output_device, self.stream)
        self.stream.synchronize()
        return output
