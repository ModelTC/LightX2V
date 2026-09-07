import unittest
from unittest.mock import patch

from lightx2v.models.runners.swiftvr.swiftvr_runner import SwiftVRRunner


class TestSwiftVRVideoWriter(unittest.TestCase):
    def test_libx265_mp4_uses_quicktime_compatible_hvc1_tag(self):
        runner = SwiftVRRunner.__new__(SwiftVRRunner)
        runner.config = {
            "video_codec": "libx265",
            "quality": 60,
            "ffmpeg_preset": "ultrafast",
        }

        with patch(
            "lightx2v.models.runners.swiftvr.swiftvr_runner.imageio.get_writer"
        ) as get_writer:
            runner.open_video_writer("output.mp4", 24)

        ffmpeg_params = get_writer.call_args.kwargs["ffmpeg_params"]
        self.assertIn("-tag:v", ffmpeg_params)
        tag_index = ffmpeg_params.index("-tag:v")
        self.assertEqual(ffmpeg_params[tag_index + 1], "hvc1")


if __name__ == "__main__":
    unittest.main()
