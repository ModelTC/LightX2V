"""Execute pinned upstream CUDA kernel bodies on CPU, or exported cases on CUDA.

The CPU mode is a serial semantic reference, never evidence of CUDA execution.
No model official implementation is imported or executed.
"""

import ctypes
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def extract(source, record):
    text = (source / record["cuda_file"]).read_text()
    name = record["cuda_kernel"]
    match = re.search(r"(?:static )?__global__ void " + name + r"\s*\(", text)
    start = text.rfind("\n", 0, match.start()) + 1
    before = text[:start].rstrip().splitlines()[-1]
    template = before if before.startswith("template<") else ""
    end = text.index("{", match.end()) + 1
    level = 1
    while level:
        level += (text[end] == "{") - (text[end] == "}")
        end += 1
    function = template + "\n" + text[match.start():end]
    if hashlib.sha256(function.encode()).hexdigest() != record["cuda_function_sha256"]:
        raise ValueError(f"Upstream CUDA kernel changed: {name}")
    return function


def build_reference(source, build, device="cpu"):
    mapping = json.loads((ROOT / "kernel-map.json").read_text())
    snapshot_revision = source / "UPSTREAM_COMMIT"
    revision = snapshot_revision.read_text().strip() if snapshot_revision.exists() else subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != mapping["upstream_commit"]:
        raise ValueError(f"Expected CuMesh {mapping['upstream_commit']}, got {revision}")
    # Packed Vec3f removes CUDA's internal padding. The three float coordinates
    # and all kernel arithmetic remain identical; exported arrays are [N,3].
    common = r'''
#include <stdint.h>
#include <cmath>
#include <algorithm>
#ifdef CPU_REFERENCE
using std::min; using std::max;
#define __global__
#define __device__
#define __host__
struct int2 {int x,y;}; struct int3 {int x,y,z;}; struct float3 {float x,y,z;};
struct Index {int x;}; static Index blockIdx{0}, blockDim{1}, threadIdx{0};
float3 make_float3(float x,float y,float z) {return {x,y,z};}
int atomicAdd(int* p,int v) {int old=*p;*p+=v;return old;}
int atomicMin(int* p,int v) {int old=*p;*p=min(*p,v);return old;}
#else
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#endif
struct Vec3f {
 float x,y,z;
 __host__ __device__ Vec3f() : x(0),y(0),z(0) {}
 __host__ __device__ Vec3f(float3 v):x(v.x),y(v.y),z(v.z) {}
 __host__ __device__ Vec3f(float a,float b,float c):x(a),y(b),z(c) {}
 __host__ __device__ Vec3f operator+(const Vec3f& b) const {return {x+b.x,y+b.y,z+b.z};}
 __device__ Vec3f operator-(const Vec3f& b) const {return {x-b.x,y-b.y,z-b.z};}
 __device__ Vec3f operator*(float a) const {return {x*a,y*a,z*a};}
 __device__ Vec3f operator/(float a) const {return {x/a,y/a,z/a};}
 __device__ float norm() const {return sqrtf(x*x+y*y+z*z);}
};
struct LessThanOp {__device__ bool operator()(float a,float b) const {return a<b;}};
'''
    text = common + "\n".join(extract(source, r) for r in mapping["kernels"])
    if device == "cuda":
        text += r'''
template<typename T>
int cub_segment_sum(const T* values,const int* offsets,T* output,int segments,cudaStream_t stream) {
    size_t bytes=0;
    auto error=cub::DeviceSegmentedReduce::Sum(nullptr,bytes,values,output,segments,offsets,offsets+1,stream);
    if(error!=cudaSuccess) return (int)error;
    void* temporary=nullptr;
    error=cudaMalloc(&temporary,bytes);
    if(error!=cudaSuccess) return (int)error;
    error=cub::DeviceSegmentedReduce::Sum(temporary,bytes,values,output,segments,offsets,offsets+1,stream);
    auto free_error=cudaFree(temporary);
    return (int)(error!=cudaSuccess ? error : free_error);
}
'''
    text += '\nextern "C" int reference(int op,void** p,int n,int arg,float threshold,void* stream) {\n'
    if device == "cpu":
        text += "for(int i=0;i<n;++i) { threadIdx.x=i; switch(op) {\n"
    else:
        text += "switch(op) {\n"
        text += "case 24: if(arg==1) return cub_segment_sum((const float*)p[0],(const int*)p[1],(float*)p[2],n,(cudaStream_t)stream); return cub_segment_sum((const Vec3f*)p[0],(const int*)p[1],(Vec3f*)p[2],n,(cudaStream_t)stream);\n"
    for record in mapping["kernels"]:
        launch = "" if device == "cpu" else "<<<(n+255)/256,256,0,(cudaStream_t)stream>>>"
        text += f"case {record['op']}: {record['cuda_kernel']}{launch}({record['launch_arguments']}); break;\n"
    text += "}\n" + ("}\nreturn 0;\n}" if device == "cpu" else "return (int)cudaGetLastError();\n}")
    build.mkdir(parents=True, exist_ok=True)
    cpp = build / ("cuda_source_reference.cpp" if device == "cpu" else "cuda_source_reference.cu")
    cpp.write_text(text)
    library = build / ("reference_" + device + ".so")
    command = ["c++", "-DCPU_REFERENCE"] if device == "cpu" else ["nvcc"]
    command += ["-std=c++17", "-O3", "-shared"]
    command += ["-fPIC"] if device == "cpu" else ["-Xcompiler", "-fPIC"]
    subprocess.run([*command, str(cpp), "-o", str(library)], check=True)
    function = ctypes.CDLL(str(library)).reference
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                         ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def run_cpu(function, op, buffers, n, arg, threshold):
    addresses = (ctypes.c_void_p * len(buffers))(*(x.ctypes.data for x in buffers))
    status = function(op, addresses, n, arg, threshold, None)
    if status:
        raise RuntimeError(f"Reference kernel {op} failed: {status}")


def verify_cuda_exports(source, export, output):
    """Run this entry on an NVIDIA machine using the exact exported MLU inputs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA hardware is required; CPU reference is a separate mode")
    function = build_reference(source, output / "build", "cuda")
    records = json.loads((export / "cases.json").read_text())
    report = []
    (output / "cuda_outputs").mkdir(exist_ok=True)
    environment = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "device": torch.cuda.get_device_name(),
                   "capability": torch.cuda.get_device_capability(),
                   "visible_device_count": torch.cuda.device_count()}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    for record in records:
        with np.load(export / record["file"], allow_pickle=False) as saved:
            buffers = [torch.from_numpy(saved[f"before_{i}"]).cuda() for i in range(record["buffer_count"])]
            torch.cuda.synchronize()
            addresses = (ctypes.c_void_p * len(buffers))(*(x.data_ptr() for x in buffers))
            error = function(record["op"], addresses, record["n"], record["arg"],
                             record["threshold"], torch.cuda.current_stream().cuda_stream)
            if error:
                raise RuntimeError(f"CUDA kernel {record['op']} launch error {error}")
            torch.cuda.synchronize()
            actual_buffers = [x.cpu().numpy() for x in buffers]
            np.savez(output / "cuda_outputs" / record["file"],
                     **{f"after_{i}": x for i, x in enumerate(actual_buffers)})
            failures, max_absolute_error = [], 0.0
            for index in record["output_indices"]:
                expected, actual = saved[f"after_{index}"], actual_buffers[index]
                try:
                    check_arrays(record["op"], index, actual, expected,
                                 [saved[f"before_{i}"] for i in range(record["buffer_count"])])
                except AssertionError as error:
                    failures.append({"buffer": index, "error": str(error)})
                if np.issubdtype(actual.dtype, np.floating) and actual.size:
                    max_absolute_error = max(max_absolute_error, float(np.abs(actual-expected).max()))
        report.append({"case": record["file"], "cuda_kernel": record["kernel"],
                       "status": "failed" if failures else "passed", "max_absolute_error": max_absolute_error,
                       "failures": failures})
    failed = sum(r["status"] != "passed" for r in report)
    result = {"reference": "actual_cuda", "device": torch.cuda.get_device_name(),
              "status": "failed" if failed else "passed", "case_count": len(report),
              "failed_cases": failed, "cases": report}
    (output / "cuda-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}), flush=True)
    return 1 if failed else 0


def check_arrays(op, index, actual, expected, before):
    if op in (2, 5) and index == (1 if op == 2 else 2):
        # CUDA atomic appends do not define order inside each CSR adjacency row.
        offsets = before[2 if op == 2 else 3]
        for a, b in zip(offsets[:-1], offsets[1:]):
            np.testing.assert_array_equal(np.sort(actual[a:b]), np.sort(expected[a:b]))
    elif op == 7:
        np.testing.assert_array_equal(np.sort(actual, axis=1), np.sort(expected, axis=1))
    elif op == 8 and index == 1:
        # Concurrent hooking can produce different intermediate parent trees.
        # The complete connected partition is checked at the pipeline boundary.
        assert np.all(actual >= 0) and np.all(actual < len(actual))
        assert np.all(actual <= np.arange(len(actual)))
    elif np.issubdtype(actual.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-7)
    else:
        np.testing.assert_array_equal(actual, expected)
