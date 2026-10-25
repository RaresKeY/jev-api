import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_lab import workspace_replay as replay


class WorkspaceReplay(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('JEV_TEST_PRIVATE_REPLAY') == '1' and replay.PLAN.exists(),
                         'Private archive integration requires original frozen source and JEV_TEST_PRIVATE_REPLAY=1')
    def test_original_calibration_metrics_and_gate_totals_reproduce(self):
        plan=replay.load_plan()
        calibration=[j for j in plan['jobs'] if j['group']=='supper-calibration']
        self.assertEqual(len(calibration),374)
        for j in calibration:
            q=next(iter(j['scoring']))
            result=replay.field_score(j['baseline_response']['answers'][q],j['scoring'][q])
            old=j['baseline_metadata']['evaluation']
            self.assertEqual(result['correct'],old['correct'])
            self.assertAlmostEqual(result['expected_mass'],old['expected_probability'],places=10)
        gates=[j for j in plan['jobs'] if j['group']=='odysseus-gates']
        self.assertEqual(sum(replay.field_score(j['baseline_response']['answers'][q],s)['correct']
                             for j in gates for q,s in j['scoring'].items()),204)

    @unittest.skipUnless(os.environ.get('JEV_TEST_PRIVATE_REPLAY') == '1' and replay.PLAN.exists(),
                         'Private archive integration requires original frozen source and JEV_TEST_PRIVATE_REPLAY=1')
    def test_every_original_repeat_gets_a_distinct_identity(self):
        plan=replay.load_plan()
        identities=[j['cloud_identity'] for j in plan['jobs']]
        self.assertEqual(len(identities),len(set(identities)))
        self.assertTrue(any(i.startswith('repeat--') for i in identities))

    def test_unknown_local_attempt_is_not_resent(self):
        with tempfile.TemporaryDirectory() as d,patch.object(replay,'DIRECTORY',Path(d)),patch.object(replay,'reuse_gates',return_value={}):
            path=Path(d)/'test'/'job.json'
            replay.save(path,dict(request_sha256='hash',status='attempted'))
            class Forbidden:
                def distributions(self,*a):raise AssertionError('Must not retry')
            with self.assertRaisesRegex(RuntimeError,'must not be retried'):
                replay.execute(Forbidden(),'test',dict(jobs=[dict(job_id='job',group='a',request_sha256='hash')]))

    def test_invalid_batch_retains_denominator_and_cannot_gain_credit(self):
        s=dict(kind='choice',expected=['correct'],label_kind='published')
        self.assertFalse(replay.field_score(None,s)['correct'])
        request=dict(questions={'a':dict(type='choice',criteria={'correct':'yes','wrong':'no'})})
        answers={'a':dict(type='choice',choice='correct',probabilities={'correct':.1,'wrong':.9})}
        with self.assertRaises(ValueError):replay.validate_answers(answers,request,strict=True)


if __name__=='__main__':unittest.main()
