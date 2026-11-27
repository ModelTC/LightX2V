"""MLU comparison reports mean latency and rejects incomplete resource evidence."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mlu_campaign

class MeanComparisonTests(unittest.TestCase):
    def test_mean_only_and_no_cross_precision_exact_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            native=root/'native';adapted=root/'pr'
            for path in (native,adapted):
                chunk=path/'chunks/000000';chunk.mkdir(parents=True)
                (chunk/'run.json').write_text('{}')
                (chunk/'resources.json').write_text(json.dumps(dict(resources_clear=True,task_cleanup_verified=True,returncode=0)))
                (path/'quality.json').write_text('{"metric": 0.5}')
            result=root/'report.json'
            def completed(path,dataset,backend):
                return [],dict(a=dict(gpu='MLU-card0',e2e_latency_s=2 if backend=='native' else 3))
            def summary(rows):return dict(mean_s=next(iter(rows.values()))['e2e_latency_s'],p95_s=1000000)
            evaluation=SimpleNamespace(MODEL='model',completed_runs=completed,latency_summary=summary,write_json=mlu_campaign.write_json)
            args=SimpleNamespace(native_run=native,lightx2v_run=adapted,dataset='data',output=result)
            with patch('builtins.print'):
                mlu_campaign.compare(evaluation,args)
            report=json.loads(result.read_text())
            self.assertEqual(report['mean_latency_ratio'],1.5)
            self.assertFalse(report['cross_precision_exact_alignment_claimed'])
            self.assertFalse(any('p95' in key.lower() for key in report))
            (adapted/'chunks/000000/resources.json').write_text(json.dumps(dict(resources_clear=False,task_cleanup_verified=True,returncode=0)))
            with self.assertRaisesRegex(ValueError,'resource'):mlu_campaign.compare(evaluation,args)

if __name__=='__main__':unittest.main()
