from loguru import logger

from lightx2v.utils.profiler import ProfilingContext4Debug


class GraphRunner:
    def __init__(self, runner):
        self.runner = runner
        self.compile()

    def compile(self):
        logger.info("start compile...")
        with ProfilingContext4Debug("compile"):
            self.runner.run_step()
        logger.info("end compile...")

    def run_pipeline(self):
        return self.runner.run_pipeline()
