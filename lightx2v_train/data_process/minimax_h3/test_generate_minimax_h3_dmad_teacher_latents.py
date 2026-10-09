import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("dmad_teacher_prep", Path(__file__).with_name("generate_minimax_h3_dmad_teacher_latents.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TeacherPreparationTests(unittest.TestCase):
    def test_padded_empty_shards_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"cache_fingerprint": "a", "condition_path": "a.pt"}]
            (root / "metadata.rank0.jsonl").write_text(json.dumps(rows[0]) + "\n")
            for rank in range(1, 8):
                (root / f"metadata.rank{rank}.jsonl").write_text("")
            self.assertEqual(module.merge_shards(root, 8, rows), rows)
            (root / "metadata.rank7.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                module.merge_shards(root, 8, rows)

    def test_filename_cannot_escape_output(self):
        name = module.teacher_filename("../../bad/fingerprint")
        self.assertEqual(Path(name).name, name)
        self.assertTrue(name.endswith(".pt"))
        self.assertEqual(module.teacher_seed("fp", 42), module.teacher_seed("fp", 42))
        self.assertNotEqual(module.teacher_seed("fp", 42), module.teacher_seed("other", 42))

    def test_main_cpu_generation_and_resume(self):
        # Actual manifest reader, condition loader, payload validation and
        # publisher; a CPU fake model substitutes only expensive H3 compute.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "source_id": "a",
                "source_row_uid": "uid",
                "cache_fingerprint": "fp",
                "condition_path": str(root / "condition.pt"),
                "caption": "test",
                "target_height": 32,
                "target_width": 64,
                "target_num_frames": 107,
            }
            positive = {
                "task": "ref2av",
                "target_height": 32,
                "target_width": 64,
                "target_num_frames": 107,
                "prompt_embeds": torch.ones(1, 3, 4),
                "text_token_tags": torch.tensor([0, 1, 1]),
                "references": [],
            }
            torch.save({"cache_fingerprint": "fp", "source_id": "a", "conditioning": {"positive": positive}}, row["condition_path"])
            metadata = root / "conditions.jsonl"
            metadata.write_text(json.dumps(row) + "\n")
            args = SimpleNamespace(conditions=metadata, model_path=root, output_dir=root / "teacher", steps=3, video_shift=12.0, audio_shift=3.0, seed=42, max_samples=None, attention_backend="native")
            latents = SimpleNamespace(video=torch.ones(1, 64, 96), audio=torch.ones(1, 356, 32))
            calls = []
            capability = SimpleNamespace(
                device=torch.device("cpu"),
                denoiser=lambda: torch.nn.Linear(1, 1),
                set_training=lambda _: None,
                encode_conditions=lambda *a: ({"condition": True}, None),
                latent_shape=lambda *a: None,
                initial_latents=lambda *a: latents,
                predict_velocity=lambda *a: calls.append(a) or latents,
                step=lambda *a: (latents, latents),
            )
            parallel = SimpleNamespace(apply=lambda _: None)
            model = SimpleNamespace(capabilities=SimpleNamespace(require=lambda kind: parallel if kind.__name__ == "ParallelCapability" else capability))
            with (
                patch.object(module, "parse_args", return_value=args),
                patch("lightx2v_train.model_zoo.build_loaded_model", return_value=model),
                patch("lightx2v_train.schedulers.flow_matching.get_device", return_value=torch.device("cpu")),
                patch("lightx2v_train.runtime.distributed.init_distributed"),
                patch("lightx2v_train.runtime.distributed.cleanup_distributed"),
            ):
                module.main()
                self.assertEqual(len(calls), 3)
                output = json.loads((args.output_dir / "metadata.jsonl").read_text())
                payload = torch.load(output["teacher_latent_path"], weights_only=True)
                self.assertEqual(payload["video"].dtype, torch.bfloat16)
                self.assertEqual(payload["source_row_uid"], "uid")
                module.main()
                self.assertEqual(len(calls), 3, "cached teacher should not be recomputed")
                args.steps = 4
                with self.assertRaisesRegex(ValueError, "recipe differs"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
