import ast
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class SeedVRSequenceParallelTest(unittest.TestCase):
    def test_dit_ops_use_seedvr_distributed_collectives(self):
        source = (REPO_ROOT / "lightx2v/models/networks/seedvr/utils/ops.py").read_text()
        distributed_source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/common/distributed/ops.py").read_text()

        self.assertIn("common.distributed.ops import", source)
        self.assertNotIn("def slice_inputs", source)
        self.assertNotIn("def gather_outputs", source)
        self.assertIn("return single_all_to_all(local_input", distributed_source)

    def test_vae_slices_before_device_transfer_and_gathers_decode_to_rank0_cpu(self):
        source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/attn_video_vae.py").read_text()
        tree = ast.parse(source)
        video_vae = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VideoAutoencoderKL")
        encode = next(node for node in video_vae.body if isinstance(node, ast.FunctionDef) and node.name == "_encode")
        calls = [ast.unparse(node) for node in ast.walk(encode) if isinstance(node, ast.Call)]

        slice_index = next(i for i, call in enumerate(calls) if call.startswith("causal_conv_slice_inputs("))
        device_index = next(i for i, call in enumerate(calls) if call == "_x.to(self.device)")
        self.assertLess(slice_index, device_index)
        self.assertIn('rank0_cpu=getattr(self, "sp_gather_decode_to_rank0", False)', source)

        context_source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/context_parallel_lib.py").read_text()
        self.assertIn("dist.recv(recv_buffer", context_source)
        self.assertIn("dist.send(x_pad", context_source)
        self.assertNotIn("dist.gather(x_pad", context_source)
        self.assertIn("def clear_causal_memory", source)
        self.assertIn("if decoded_tile is None:", source)
        self.assertIn("if not self.sp_gather_decode_to_rank0 and result.device != z.device:", source)

    def test_causal_cache_uses_neighbor_ranks(self):
        source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/context_parallel_lib.py").read_text()
        inflation_source = (REPO_ROOT / "lightx2v/models/video_encoders/hf/seedvr/causal_inflation_lib.py").read_text()

        self.assertIn("get_next_sequence_parallel_rank()", source)
        self.assertIn("get_prev_sequence_parallel_rank()", source)
        self.assertNotIn("send_dst = 0", source)
        self.assertNotIn("recv_src = 0", source)
        self.assertIn("cache_owner = self.sp_cache_owner_index % sp_size", inflation_source)
        self.assertIn("dist.recv(self.memory, last_global_rank", inflation_source)

    def test_sp_config_enables_tiling_and_keeps_bounded_causal_slices(self):
        config = json.loads((REPO_ROOT / "configs/seedvr/seedvr2_7b_sp.json").read_text())

        self.assertTrue(config["use_tiling_vae"])
        self.assertTrue(config["stream_save_video"])
        self.assertEqual(config["vae_causal_slice_size"], 4)
        self.assertGreater(config["parallel"]["seq_p_size"], 1)
        self.assertEqual(config["parallel"]["seq_p_attn_type"], "ulysses")
        self.assertTrue(config["parallel"]["vae_parallel"])

        runner_source = (REPO_ROOT / "lightx2v/models/runners/seedvr/seedvr_runner.py").read_text()
        self.assertNotIn("VAE sequence parallel requires use_tiling_vae=false", runner_source)
        self.assertIn("video_recorder.pub_video(video)", runner_source)
        self.assertIn("video_recorder.stop(wait=False)", runner_source)

        recorder_source = (REPO_ROOT / "lightx2v/utils/video_recorder.py").read_text()
        self.assertIn("self.video_conn.sendall(frame.tobytes())", recorder_source)

    def test_sp_launcher_uses_torchrun(self):
        source = (REPO_ROOT / "scripts/seedvr2/run_seedvr2_7b_sp.sh").read_text()

        self.assertIn("torchrun --nproc_per_node=", source)
        self.assertIn("seedvr2_7b_sp.json", source)


if __name__ == "__main__":
    unittest.main()
