import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.prepare_wan_vae_metadata import prepare_metadata
from lightx2v_train.data.video_dataset import VideoDataset
from lightx2v_train.model_zoo.wan.vae_data_process import Wan21VAEDistillationProcessor


class WanMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.videos = self.root / "remote/videos"
        self.videos.mkdir(parents=True)
        self.source = self.root / "metadata.jsonl"
        self.output = self.root / "metadata_vae.jsonl"
        self.row = dict(video="videos/000001.mp4", prompt_path="prompts/000001.txt",
                        width=2048, height=2048, frames=201, fps=16.0)
        self.source.write_text(json.dumps(self.row) + "\n")

    def test_rewritten_manifest_loads_without_prompts_and_preserves_geometry(self):
        video = self.videos / "000001.mp4"
        video.touch()
        self.assertEqual(prepare_metadata(self.source, self.output, self.videos, check_files=True), 1)
        rewritten = json.loads(self.output.read_text())
        self.assertEqual(rewritten, {**{k: v for k, v in self.row.items() if k != "prompt_path"}, "video": str(video)})
        self.assertEqual(json.loads(self.source.read_text()), self.row)
        dataset = VideoDataset(self.output, skip_missing=False, sample_processor=Wan21VAEDistillationProcessor(65))
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.samples[0]["meta"]["video_path"], str(video))

    def test_existing_output_is_not_overwritten(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare_metadata(self.source, self.output, self.videos)
        self.assertEqual(self.output.read_text(), "keep")

    def test_missing_video_leaves_no_partial_manifest(self):
        with self.assertRaises(FileNotFoundError):
            prepare_metadata(self.source, self.output, self.videos, check_files=True)
        self.assertFalse(self.output.exists())

    def test_path_outside_video_root_leaves_no_manifest(self):
        self.source.write_text(json.dumps({**self.row, "video": "videos/../outside.mp4"}))
        with self.assertRaises(ValueError):
            prepare_metadata(self.source, self.output, self.videos)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
