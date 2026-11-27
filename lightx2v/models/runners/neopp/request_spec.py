from lightx2v.models.runners.request_args import COMMON, without
from lightx2v.utils.request_spec import Argument, MultiTaskSpec


class NeoppSpec(MultiTaskSpec):
    def get_cli_arguments(self, config, task):
        cfg = config.get("enable_cfg", False)
        return dict(
            super().get_cli_arguments(config, task),
            cond_kv=Argument(input_info_default="", help="Conditioning KV cache captured from LightLLM.", required=True),
            uncond_kv=Argument(input_info_default="", help="Unconditional KV cache; required when CFG is enabled.", required=cfg),
            index_offset_cond=Argument(type=int, help="Conditioning position offset from the captured request.", required=True),
            index_offset_uncond=Argument(type=int, help="Unconditional position offset from the captured request.", required=cfg),
        )

    def prepare_runner(self, runner, args):
        runner.load_kvcache(args.cond_kv, args.uncond_kv)
        options = {name: runner.config[name] for name in ("cfg_interval", "cfg_scale", "cfg_norm", "timestep_shift") if name in runner.config}
        runner.set_inference_params(args.index_offset_cond, args.index_offset_uncond, **options)


NEO_IMAGE = dict(
    without(COMMON, "return_result_tensor"),
    size=Argument(input_info_default=[], type=list[int], help="Output size in pixels: HEIGHT WIDTH"),
    num_loops=Argument(type=int, ge=1, le=16, strict=True, help="Override the checkpoint's generation core passes per denoising step (1-16)."),
)

SPECS = {"neopp": NeoppSpec({"t2i": NEO_IMAGE, "i2i": NEO_IMAGE})}
