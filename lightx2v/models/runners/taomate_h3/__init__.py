__all__ = ["TaoMateH3Runner"]


def __getattr__(name):
    if name == "TaoMateH3Runner":
        from .runner import TaoMateH3Runner

        globals()[name] = TaoMateH3Runner
        return TaoMateH3Runner
    raise AttributeError(name)
