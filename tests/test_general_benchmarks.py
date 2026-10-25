import unittest
from importlib.util import find_spec
from scripts.general_benchmarks import numeric_answer,score_ifeval

class GeneralBenchmarks(unittest.TestCase):
    def test_numeric_answer_matches_original_marker_and_normalizes(self):
        self.assertEqual(numeric_answer('Work is irrelevant.\n#### 1,200.00'),'1.2E+3')
        self.assertEqual(numeric_answer('#### -2.50'),'-2.5')
        self.assertIsNone(numeric_answer('The answer is 1200.'))
        self.assertEqual(numeric_answer('#### 10\ncorrection: #### 12'),'12')

    @unittest.skipUnless(all(find_spec(name) for name in ('absl', 'nltk', 'langdetect', 'immutabledict')),
                         'IFEval requires the optional packages in requirements-benchmarks.txt')
    def test_ifeval_original_keyword_validator(self):
        case={'original':dict(key=1,prompt='Include hello.',instruction_id_list=['keywords:existence'],kwargs=[{'keywords':['hello']}])}
        self.assertTrue(score_ifeval(case,'hello')['prompt_strict'])
        self.assertFalse(score_ifeval(case,'goodbye')['prompt_strict'])
        self.assertFalse(score_ifeval(case,'')['prompt_loose'])

if __name__=='__main__':unittest.main()
