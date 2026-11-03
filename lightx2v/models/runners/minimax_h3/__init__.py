__all__ = ["MiniMaxH3Runner"]


def __getattr__(name):
    if name == "MiniMaxH3Runner":
        from .minimax_h3_runner import MiniMaxH3Runner

        globals()[name] = MiniMaxH3Runner
        return MiniMaxH3Runner
    raise AttributeError(name)
