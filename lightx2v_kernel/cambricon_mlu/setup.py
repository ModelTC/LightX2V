from setuptools import find_packages, setup
from torch_mlu.utils.cpp_extension import BuildExtension, MLUExtension

setup(
    name="lightx2v-kernel-mlu",
    version="0.1.0",
    packages=find_packages("python"),
    package_dir={"": "python"},
    ext_modules=[
        MLUExtension(
            "lightx2v_kernel_mlu._C",
            sources=["csrc/bindings.cpp", "csrc/mesh_kernels.mlu"],
            extra_compile_args={"cxx": ["-O3"], "cncc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    license="MIT",
)
