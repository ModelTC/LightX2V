import functools
from typing import Dict, List, Optional

import torch
from loguru import logger


def compiled_method(compile_options: Optional[Dict] = None):
    def decorator(func):
        func_name = func.__name__
        compile_opts = compile_options or {}

        state = {
            "original_func": func,
            "compiled_graphs": {},
            "compile_mode": False,
            "selected_graph": None,
            "selected_compiled": None,
        }

        @functools.wraps(func)
        def wrapper(self, *args, graph_name: Optional[str] = None, **kwargs):
            if state["compile_mode"]:
                if graph_name is None:
                    graph_name = f"graph_{len(state['compiled_graphs']) + 1:02d}"

                if graph_name not in state["compiled_graphs"]:
                    logger.info(f"[Compile] Compiling {func_name} as '{graph_name}'...")

                    compiled_func = torch.compile(state["original_func"], **compile_opts)

                    try:
                        result = compiled_func(self, *args, **kwargs)
                        state["compiled_graphs"][graph_name] = compiled_func
                        logger.info(f"[Compile] Compiled {func_name} as '{graph_name}'")
                        return result
                    except Exception as e:
                        logger.info(f"[Compile] Failed to compile {func_name} as '{graph_name}': {e}")
                        return state["original_func"](self, *args, **kwargs)
                else:
                    logger.info(f"[Compile] Using existing compiled graph '{graph_name}'")
                    return state["compiled_graphs"][graph_name](self, *args, **kwargs)

            elif state["selected_compiled"]:
                return state["selected_compiled"](self, *args, **kwargs)
            else:
                return state["original_func"](self, *args, **kwargs)

        def _enable_compile_mode():
            logger.info(f"[Compile] Enabling compile mode for {func_name}")
            state["compile_mode"] = True

        def _disable_compile_mode():
            logger.info(f"[Compile] Disabling compile mode for {func_name}")
            state["compile_mode"] = False

        def _select_graph(graph_name: str):
            if graph_name not in state["compiled_graphs"]:
                logger.warning(f"[Compile] Graph '{graph_name}' not found. Available graphs: {list(state['compiled_graphs'].keys())}, returning to original function.")
                state["selected_graph"] = None
                state["selected_compiled"] = None
            else:
                logger.info(f"[Compile] Selecting graph '{graph_name}' for {func_name}")
                state["selected_graph"] = graph_name
                state["selected_compiled"] = state["compiled_graphs"][graph_name]
                logger.info(f"[Compile] {func_name} will now use graph '{graph_name}' for inference")

        def _unselect_graph():
            logger.info(f"[Compile] Unselecting graph for {func_name}, returning to original function")
            state["selected_graph"] = None
            state["selected_compiled"] = None

        def _get_status():
            return {
                "available_graphs": list(state["compiled_graphs"].keys()),
                "compiled_count": len(state["compiled_graphs"]),
                "selected_graph": state["selected_graph"],
                "compile_mode": state["compile_mode"],
                "mode": "compile" if state["compile_mode"] else ("inference" if state["selected_compiled"] else "original"),
            }

        def _clear_graphs():
            state["compiled_graphs"].clear()
            state["selected_graph"] = None
            state["selected_compiled"] = None
            state["compile_mode"] = False
            logger.info(f"[Compile] Cleared all compiled graphs for {func_name}")

        def _remove_graph(graph_name: str):
            if graph_name in state["compiled_graphs"]:
                del state["compiled_graphs"][graph_name]
                if state["selected_graph"] == graph_name:
                    state["selected_graph"] = None
                    state["selected_compiled"] = None
                logger.info(f"[Compile] Removed graph '{graph_name}' for {func_name}")
            else:
                logger.info(f"[Compile] Graph '{graph_name}' not found")

        wrapper._enable_compile_mode = _enable_compile_mode
        wrapper._disable_compile_mode = _disable_compile_mode
        wrapper._select_graph = _select_graph
        wrapper._unselect_graph = _unselect_graph
        wrapper._get_status = _get_status
        wrapper._clear_graphs = _clear_graphs
        wrapper._remove_graph = _remove_graph
        wrapper._func_name = func_name

        return wrapper

    return decorator


class CompiledMethodsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compiled_methods = {}
        self._discover_compiled_methods()

    def _discover_compiled_methods(self):
        logger.info(f"[Compile] Discovering compiled methods for {self.__class__.__name__}...")

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_enable_compile_mode"):
                logger.info(f"[Compile] Found compiled method: {attr_name}")
                self._compiled_methods[attr_name] = attr

    def enable_compile_mode(self, method_name: str = None):
        if method_name:
            if method_name not in self._compiled_methods:
                raise ValueError(f"Method '{method_name}' is not a compiled method")
            self._compiled_methods[method_name]._enable_compile_mode()
        else:
            for name, method in self._compiled_methods.items():
                method._enable_compile_mode()
            logger.info("[Compile] Enabled compile mode for all methods")

    def disable_compile_mode(self, method_name: str = None):
        if method_name:
            if method_name not in self._compiled_methods:
                raise ValueError(f"Method '{method_name}' is not a compiled method")
            self._compiled_methods[method_name]._disable_compile_mode()
        else:
            for name, method in self._compiled_methods.items():
                method._disable_compile_mode()
            logger.info("[Compile] Disabled compile mode for all methods")

    def select_graph(self, method_name: str, graph_name: str):
        if method_name not in self._compiled_methods:
            raise ValueError(f"Method '{method_name}' is not a compiled method")

        method = self._compiled_methods[method_name]
        method._select_graph(graph_name)

    def unselect_graph(self, method_name: str):
        if method_name not in self._compiled_methods:
            raise ValueError(f"Method '{method_name}' is not a compiled method")

        method = self._compiled_methods[method_name]
        method._unselect_graph()

    def get_compile_status(self) -> Dict:
        status = {}
        for method_name, method in self._compiled_methods.items():
            status[method_name] = method._get_status()
        return status

    def get_compiled_methods(self) -> List[str]:
        return list(self._compiled_methods.keys())

    def clear_compiled_graphs(self, method_name: str = None):
        if method_name:
            if method_name in self._compiled_methods:
                self._compiled_methods[method_name]._clear_graphs()
            else:
                logger.info(f"Method '{method_name}' not found")
        else:
            for method_name, method in self._compiled_methods.items():
                method._clear_graphs()
            logger.info("[Compile] Cleared all compiled graphs")

    def remove_graph(self, method_name: str, graph_name: str):
        if method_name not in self._compiled_methods:
            raise ValueError(f"Method '{method_name}' is not a compiled method")

        method = self._compiled_methods[method_name]
        method._remove_graph(graph_name)
