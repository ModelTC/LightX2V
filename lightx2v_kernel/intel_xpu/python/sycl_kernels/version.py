from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sycl-kernels")
except PackageNotFoundError:
    __version__ = "0.0.1"
