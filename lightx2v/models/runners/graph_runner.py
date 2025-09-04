from loguru import logger

from lightx2v.utils.profiler import *


class GraphRunner:
    def __init__(self, runner):
        self.runner = runner
        self.compile()

    def compile(self):
        logger.info("=" * 60)
        logger.info("🚀 Starting Model Compilation - Please wait, this may take a while... 🚀")
        logger.info("=" * 60)

        with ProfilingContext4DebugL2("compile"):
            self.runner.run_step()

        logger.info("=" * 60)
        logger.info("✅ Model Compilation Completed ✅")
        logger.info("=" * 60)

    def run_pipeline(self):
        return self.runner.run_pipeline()
