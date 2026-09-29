from simulator.libero_node.bench.adapter import LiberoAdapter


class LiberoPlusAdapter(LiberoAdapter):
    """Same simulator contract; perturbation metadata and init states come from Plus.

    Kept separate so benchmark-specific behavior never leaks into plain LIBERO.
    """

    def __init__(self, cfg, task):
        if not task.get("category"):
            raise ValueError("Every LIBERO-plus task must have a perturbation category")
        super().__init__(cfg, task)
