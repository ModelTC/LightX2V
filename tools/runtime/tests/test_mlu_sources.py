"""Native device overlays must reconstruct exactly and reject unrelated edits."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlu_sources

class SourceOverlayTests(unittest.TestCase):
    def test_reconstruction_hashes_and_unexpected_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            subprocess.run(['git','init','--quiet',str(root)],check=True)
            source=root/'model.py';source.write_text('device = "cuda"\n')
            subprocess.run(['git','-C',str(root),'add','model.py'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid','commit','--quiet','-m','base'],check=True)
            spec=dict(repository='unused',revision=mlu_sources.git(root,'rev-parse','HEAD').decode().strip(),mlu_patches=[dict(path='model.py',sha256=hashlib.sha256(source.read_bytes()).hexdigest(),runtime_sha256=hashlib.sha256(b'device = "mlu"\n').hexdigest(),replacements=[dict(before='"cuda"',after='"mlu"')])])
            mlu_sources.prepare(root,spec);mlu_sources.verify(root,spec)
            mlu_sources.prepare(root,spec)
            self.assertEqual(source.read_text(),'device = "mlu"\n')
            (root/'unexpected.py').write_text('x=1\n')
            with self.assertRaisesRegex(ValueError,'Unexpected'): mlu_sources.verify(root,spec)
            (root/'unexpected.py').unlink()
            source.write_text('device = "cpu"\n')
            with self.assertRaisesRegex(ValueError,'Refusing'): mlu_sources.prepare(root,spec)

if __name__=='__main__':unittest.main()
