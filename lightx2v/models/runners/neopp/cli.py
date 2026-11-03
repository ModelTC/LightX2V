from lightx2v.models.runners.request_args import COMMON, fields, without
from lightx2v.utils.request_spec import Argument, MultiTaskSpec


class NeoppSpec(MultiTaskSpec):
    def get_cli_arguments(self, config, task):
        cfg = config.get("enable_cfg", False)
        return super().get_cli_arguments(config, task) | {
            "cond_kv": Argument(help="Conditioning KV cache captured from LightLLM.", required=True),
            "uncond_kv": Argument(help="Unconditional KV cache; required when CFG is enabled.", required=cfg),
            "index_offset_cond": Argument(type=int, help="Conditioning position offset from the captured request.", required=True),
            "index_offset_uncond": Argument(type=int, help="Unconditional position offset from the captured request.", required=cfg),
        }

    def prepare_runner(self, runner, args):
        runner.load_kvcache(args.cond_kv, args.uncond_kv)
        options = {name: runner.config[name] for name in ("cfg_interval", "cfg_scale", "cfg_norm", "timestep_shift") if name in runner.config}
        runner.set_inference_params(args.index_offset_cond, args.index_offset_uncond, **options)


SPECS = {
    "neopp": NeoppSpec(
        {
            "t2i": without(COMMON, "return_result_tensor") | fields("size"),
            "i2i": without(COMMON, "return_result_tensor") | fields("size"),
        },
    ),
}
