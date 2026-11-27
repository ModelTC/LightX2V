"""Keep the validated MLU evaluator distinct from the CUDA source lock."""
import json
from pathlib import Path
import unittest

class MLUGenEvalSourceLock(unittest.TestCase):
    def test_known_mlu_runtime_and_reproducible_upstream_base(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        mlu=json.loads((root/'mlu/source-lock.json').read_text())['geneval']
        cuda=json.loads((root/'nvidia/source-lock.json').read_text())['geneval']
        self.assertEqual(mlu['revision'],cuda['revision'])
        patch=mlu['mlu_patches'][0]
        self.assertEqual(patch['path'],'evaluation/evaluate_images.py')
        self.assertEqual(patch['sha256'],cuda['files'][0]['sha256'])
        self.assertEqual(patch['runtime_sha256'],'f767660eeee7d2fc399912dbd0263c1cd9f945cb2b14bb4676054ee8a317f827')
        self.assertNotEqual(patch['runtime_sha256'],patch['sha256'])
        self.assertGreater(len(patch['replacements']),0)

if __name__=='__main__':unittest.main()
