import math,unittest
from importlib.util import find_spec
from types import SimpleNamespace
from decision_lab.benchmarks import score_record,summarize_rows,partial,validate_record
from vendored.jevbench.scoring import score_task
from scripts.upstream_benchmarks import evaluate

class UpstreamBenchmarks(unittest.TestCase):
    def case(self,kind='choice',expected='b'):
        q=dict(type=kind,instructions='Select the answer.',criteria={'a':'A','b':'B'} if kind=='choice' else ['low','medium','high'])
        return dict(suite='test',id='test/1',questions={'decision':q},expected={'decision':expected},stratum='x')

    def test_exact_upstream_distribution_scoring(self):
        r=self.case()
        for p in [{'a':.2,'b':.8},{'a':.5,'b':.5},{'a':.2,'b':.799},{'a':.2,'b':.9},{'a':True,'b':0},{'a':float('nan'),'b':0},{}]:
            got=score_record(r,{'decision':p})['decision']
            original=score_task(p,SimpleNamespace(labels=['a','b'],expected='b',question=r['questions']['decision']))
            for key,value in original.items():self.assertEqual(got[key],value)

    def test_score_accuracy_is_mode_not_workstation_tolerance(self):
        r=self.case('score','1');p={'0':.4,'1':.3,'2':.3}
        f=score_record(r,{'decision':p})['decision']
        self.assertFalse(f['correct']);self.assertAlmostEqual(f['ordinal_ev'],.9)

    def test_soft_gold_kl_and_brier(self):
        r=self.case();r['gold']={'decision':{'a':.25,'b':.75}}
        f=score_record(r,{'decision':{'a':.5,'b':.5}})['decision']
        self.assertAlmostEqual(f['soft_brier'],.125)
        self.assertAlmostEqual(f['kl_from_gold'],.25*math.log(.5)+.75*math.log(1.5))

    def test_label_only_does_not_invent_calibration(self):
        f=score_record(self.case(),predictions={'decision':'b'})
        summary=summarize_rows([dict(fields=f)])
        self.assertEqual(summary['accuracy'],1);self.assertIsNone(summary['hard_brier']);self.assertIsNone(summary['ece'])

    def test_failures_stay_in_accuracy_denominator(self):
        r=self.case();good=score_record(r,predictions={'decision':'b'});bad=score_record(r,predictions={})
        self.assertEqual(summarize_rows([dict(fields=good),dict(fields=bad)])['accuracy'],.5)

    def test_only_inputs_reach_backend(self):
        class Capture:
            last_raw={}
            def distributions(self,state,questions):
                self.inputs=(state,questions)
                return {'decision':{'a':.1,'b':.9}}
        c=Capture();r=self.case();r.update(state='Original text',original={'secret_gold':'b'},gold={'decision':{'a':0,'b':1}})
        evaluate(c,r)
        self.assertEqual(c.inputs,(r['state'],r['questions']))
        self.assertNotIn('label',c.inputs[1]['decision'])

    def test_sampling_is_deterministic_stratified_and_input_order_independent(self):
        rows=[dict(id=str(i),stratum=str(i%3)) for i in range(30)]
        self.assertEqual(partial(rows,9),partial(list(reversed(rows)),9))
        self.assertEqual({k:sum(r['stratum']==k for r in partial(rows,9)) for k in ['0','1','2']},{'0':3,'1':3,'2':3})

    def test_invalid_source_gold_rejected(self):
        r=self.case();r['gold']={'decision':{'a':.4,'b':.4}}
        with self.assertRaises(ValueError):validate_record(r)


class NativeAndDerived(unittest.TestCase):
    def test_scalar_head_adapter_preserves_fractional_expected_scores(self):
        from decision_lab.specialists import TypedDistributionClient
        class Head(TypedDistributionClient):
            def distributions(self,state,questions):
                return {'rating':{'0':.4,'1':.3,'2':.3},'flag':{'false':.2,'true':.8}}
        qs={'rating':dict(type='score',instructions='Rate it',criteria=['low','mid','high']), 'flag':dict(type='noul',instructions='Is it true?')}
        values=Head().evaluate('state',qs)
        self.assertAlmostEqual(values['rating'],.9);self.assertAlmostEqual(values['flag'],.8)

    @unittest.skipUnless(find_spec('torch'), 'NLI tensor integration requires optional PyTorch')
    def test_native_nli_uses_model_label_names(self):
        import torch
        from decision_lab.nli import NLIClient
        c=NLIClient.__new__(NLIClient);c.torch=torch;c.device='cpu'
        c.model=SimpleNamespace(config=SimpleNamespace(id2label={0:'contradiction',1:'entailment',2:'neutral'}))
        c._logits=lambda pairs:torch.tensor([[0.,10.,-1.]])
        p=c.pair_distributions([('premise','hypothesis')])[0]
        self.assertGreater(p['entailment'],.99);self.assertLess(p['contradiction'],.01)

    def test_option_order_probe_preserves_source_and_gold(self):
        from decision_lab.benchmark_diagnostics import option_order_probe
        c=dict(id='test/x',state='Original state',questions={'q':dict(type='choice',instructions='Choose',criteria={'a':'A','b':'B'})},expected={'q':'b'},gold={'q':{'a':.2,'b':.8}})
        p=option_order_probe(c)
        self.assertEqual(list(p['questions']['q']['criteria']),['b','a'])
        self.assertEqual(p['expected'],c['expected']);self.assertEqual(p['gold'],c['gold']);self.assertEqual(p['state'],c['state'])
        self.assertEqual(list(c['questions']['q']['criteria']),['a','b'])

    def test_native_kev_clean_population_matches_original_math(self):
        from decision_lab.benchmark_diagnostics import kev_native
        from vendored.kev.metrics import metrics
        c=dict(suite='kev-transfer',id='test/1',stratum='source',questions={'q':dict(type='choice',criteria={'a':'A','b':'B'})},expected={'q':'b'},original={'_meta':{'variant':'clean'}})
        r=dict(case=c,fields={'q':dict(valid=True,probs={'a':.2,'b':.8})})
        self.assertEqual(kev_native([r])['micro'],metrics([dict(p=[.2,.8],label=1,type='choice')]))
        r['fields']['q']['valid']=False
        self.assertEqual(kev_native([r])['status'],'incomplete-native-population')

if __name__=='__main__':unittest.main()
