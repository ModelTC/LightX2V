import os
from pathlib import Path
from subprocess import Popen

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from cuda import cudart
from loguru import logger

from lightx2v.common.backend_infer.trt import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class HyVaeTrtModelInfer(nn.Module):
    """
    Implements inference for the TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        # Load TRT engine
        if not Path(engine_path).exists():
            # dir_name = str(Path(engine_path).parents)
            # onnx_path = self.export_to_onnx(decoder, dir_name)
            # self.convert_to_trt_engine(onnx_path, engine_path)
            raise FileNotFoundError(f"VAE tensorrt engine `{str(engine_path)}` not exists.")
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context
        logger.info(f"Loaded VAE tensorrt engine from `{engine_path}`")

    def alloc(self, shape_dict):
        """
        Setup I/O bindings
        """
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            # shape = self.engine.get_tensor_shape(name)
            shape = shape_dict[name]
            if is_input:
                self.context.set_input_shape(name, shape)
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = common.cuda_call(cudart.cudaMalloc(size))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]

    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def __call__(self, batch, top=1):
        """
        Execute inference
        """
        # Prepare the output data
        device = batch.device
        dtype = batch.dtype
        batch = batch.cpu().numpy()

        def get_output_shape(shp):
            b, c, t, h, w = shp
            out = (b, 3, 4 * (t - 1) + 1, h * 8, w * 8)
            return out

        shp_dict = {"inp": batch.shape, "out": get_output_shape(batch.shape)}
        self.alloc(shp_dict)
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        common.memcpy_host_to_device(self.inputs[0]["allocation"], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        common.memcpy_device_to_host(output, self.outputs[0]["allocation"])
        output = torch.from_numpy(output).to(device).type(dtype)
        return output

    @staticmethod
    def export_to_onnx(decoder: torch.nn.Module, model_dir):
        logger.info("Start to do VAE onnx exporting.")
        device = next(decoder.parameters())[0].device
        example_inp = torch.rand(1, 16, 17, 32, 32).to(device).type(next(decoder.parameters())[0].dtype)
        out_path = str(Path(str(model_dir)) / "vae_decoder.onnx")
        torch.onnx.export(
            decoder.eval().half(),
            example_inp.half(),
            out_path,
            input_names=["inp"],
            output_names=["out"],
            opset_version=14,
            dynamic_axes={"inp": {1: "c1", 2: "c2", 3: "c3", 4: "c4"}, "out": {1: "c1", 2: "c2", 3: "c3", 4: "c4"}},
        )
        # onnx_ori = onnx.load(out_path)
        os.system(f"onnxsim {out_path} {out_path}")
        # onnx_opt, check = simplify(onnx_ori)
        # assert check, f"Simplified ONNX model({out_path}) could not be validated."
        # onnx.save(onnx_opt, out_path)
        logger.info("Finish VAE onnx exporting.")
        return out_path

    @staticmethod
    def convert_to_trt_engine(onnx_path, engine_path):
        logger.info("Start to convert VAE ONNX to tensorrt engine.")
        cmd = (
            "trtexec "
            f"--onnx={onnx_path} "
            f"--saveEngine={engine_path} "
            "--allowWeightStreaming "
            "--stronglyTyped "
            "--fp16 "
            "--weightStreamingBudget=100 "
            "--minShapes=inp:1x16x9x18x16 "
            "--optShapes=inp:1x16x17x32x16 "
            "--maxShapes=inp:1x16x17x32x32 "
        )
        p = Popen(cmd, shell=True)
        p.wait()
        if not Path(engine_path).exists():
            raise RuntimeError(f"Convert vae onnx({onnx_path}) to tensorrt engine failed.")
        logger.info("Finish VAE tensorrt converting.")
        return engine_path
