from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_workspace_replay import owner_error,divergence


class OriginalInterpretation(unittest.TestCase):
    def test_application_rejects_inconsistent_choice_without_repair(self):
        answer=dict(type='choice',choice='yes',probabilities={'yes':.1,'no':.9},confidence=.8)
        for group in ('supper-drone-messages','task-routing','browser-decisions','evo-probe'):
            self.assertEqual(owner_error(dict(group=group),{'q':answer}),'OwnerChoiceMismatch')
            self.assertEqual(answer['choice'],'yes')
        # Original public eval records explicit-choice mismatches as diagnostics.
        self.assertIsNone(owner_error(dict(group='anli'),{'q':answer}))

    def test_original_score_consistency_and_invalid_confidence(self):
        answer=dict(type='score',score=0.,confidence=.8,probabilities={'0':.1,'1':.9})
        self.assertEqual(owner_error(dict(group='supper-calibration'),{'q':answer}),'OwnerScoreMismatch')
        answer.update(score=.9,confidence=float('nan'))
        self.assertEqual(owner_error(dict(group='supper-calibration'),{'q':answer}),'OwnerConfidenceInvalid')

    def test_distribution_distance_uses_normalized_rounding_only(self):
        self.assertEqual(divergence({'a':.99,'b':0.},{'a':1.,'b':0.}),0)
        self.assertAlmostEqual(divergence({'a':1.,'b':0.},{'a':0.,'b':1.}),1)


if __name__=='__main__':unittest.main()
