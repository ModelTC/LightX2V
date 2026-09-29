from importlib import import_module


def get_benchmark_backend(benchmark):
    modules = {
        "libero": "simulator.libero_node.bench.config",
        "libero_plus": "simulator.libero_node.bench.config",
        "robotwin": "simulator.robotwin_node.bench.config",
    }
    if benchmark not in modules:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return import_module(modules[benchmark])
