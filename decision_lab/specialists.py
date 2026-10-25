"""Local, non-generative runtimes for custom decision checkpoints.

Upstream math/formatting is vendored; no remote code or model downloads at runtime.
FP32 is explicit because the RTX 2080 Ti lacks native BF16 support.
"""
import json
from pathlib import Path


def labels(question):
    if question['type']=='noul':return ['false','true']
    if question['type']=='choice':return list(question['criteria'])
    return [str(i) for i in range(len(question['criteria']))]


class TypedDistributionClient:
    def evaluate(self,state,questions,examples=()):
        from .contracts import validate
        if examples:raise ValueError('Dedicated decision heads do not consume chat demonstrations')
        distributions=self.distributions(state,questions)
        values={}
        for name,q in questions.items():
            p=distributions[name]
            if q['type']=='noul':values[name]=p['true']
            elif q['type']=='choice':values[name]=max(p,key=p.get)
            else:values[name]=sum(int(k)*v for k,v in p.items())
        return validate(values,questions,fractional_scores=True)


class KevClient(TypedDistributionClient):
    def __init__(self,path,base):
        import torch
        from transformers import AutoTokenizer
        from peft import PeftModel
        from vendored.kev.model import DecisionModel
        self.torch=torch;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        path,base=Path(path),Path(base)
        meta=torch.load(path/'head.pt',map_location='cpu',weights_only=True)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=DecisionModel(str(base),self.tokenizer,self.device,dtype=torch.float32,head_dim=meta.get('head_dim',256))
        self.model.lm=PeftModel.from_pretrained(self.model.lm,str(path),local_files_only=True).to(self.device).merge_and_unload()
        self.model.head.load_state_dict(meta['head'],strict=True)
        self.model.head.temperature=meta.get('temperature',1.)
        self.model.eval();self.last_raw={}

    def distributions(self,state,questions):
        from vendored.kev.api import SystemOneRequest,to_record
        req=SystemOneRequest(state=state,questions=questions)
        record,meta=to_record(req)
        enc=self.model.encode(self.tokenizer,record,max_state=65536,max_branch=73728,strict=True)
        with self.torch.inference_mode():probs=self.model.probs(enc)
        result={m['id']:dict(zip(m['keys'],map(float,p))) for m,p in zip(meta,probs)}
        self.last_raw=dict(distributions=result,device=self.device,dtype='float32',temperature=self.model.head.temperature,state_tokens=enc['state_tokens'],state_truncated=enc['state_truncated'])
        return result


class LayaClient(TypedDistributionClient):
    def __init__(self,path):
        import torch
        from transformers import PreTrainedTokenizerFast
        from safetensors.torch import load_file
        from vendored.laya.common import build_model
        path=Path(path);self.cfg=json.loads((path/'rl_agent_config.json').read_text())
        self.torch=torch;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer=PreTrainedTokenizerFast.from_pretrained(path/'tokenizer',local_files_only=True)
        self.model=build_model(self.cfg,encoder_dir=str(path/'encoder'),pretrained=False)
        self.model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
        self.model.to(self.device).eval();self.last_raw={}

    def distributions(self,state,questions):
        from vendored.laya.common import build_sequence,collate_items,QTYPES,temp_bucket
        items=[];stats={}
        for name,q in questions.items():
            internal=dict(t=q['type'],ins=q['instructions'],crit=q.get('criteria'))
            ids,markers,option_stats,trunc=build_sequence(self.tokenizer,state,internal,max_len=self.cfg['max_len'],head_max_len=self.cfg['head_max_len'],return_stats=True,return_truncation_stats=True)
            if len(markers)!=len(labels(q)):raise ValueError('Laya lost an option marker')
            items.append(dict(ids=ids,markers=markers,qtype=QTYPES[q['type']]))
            stats[name]=dict(options=option_stats,truncation=trunc)
        batch=collate_items([items],self.tokenizer.pad_token_id)
        inputs={k:v.to(self.device) for k,v in batch.items() if k in {'input_ids','attention_mask','marker_pos','marker_mask','qtype'}}
        with self.torch.inference_mode():logits,_=self.model(**inputs)
        result={}
        for i,(name,q) in enumerate(questions.items()):
            keys=labels(q);bucket=temp_bucket(QTYPES[q['type']],len(keys))
            temperature=self.cfg.get('temperature_by_options',{}).get(bucket,self.cfg['temperature'][QTYPES[q['type']]])
            probs=self.torch.softmax(logits[i,:len(keys)]/temperature,-1).cpu().tolist()
            result[name]=dict(zip(keys,probs));stats[name]['temperature']=temperature
        self.last_raw=dict(distributions=result,stats=stats,device=self.device,dtype='float32',act_head='computed; escalation policy not applied')
        return result


class VerdictClient(TypedDistributionClient):
    def __init__(self,path):
        import torch
        from gliclass import GLiClassModel
        from transformers import AutoTokenizer
        from vendored.verdict.core.engine_encoder import DecisionEngine
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        model=GLiClassModel.from_pretrained(str(path),local_files_only=True).to(self.device).eval()
        tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        self.engine=DecisionEngine(model_name_or_path=str(path),model=model,tokenizer=tokenizer,device=self.device)
        self.last_raw={}

    def candidate_distributions(self,text,question,candidates):
        """Original Verdict evaluator path: candidate order and abstention position unchanged."""
        import torch
        from vendored.verdict.core.formatting import build_model_input
        keys=[c['id'] for c in candidates]
        if len(keys)>25:raise ValueError('Verdict capacity exceeded: at most 25 original candidates')
        prompt=build_model_input(question,text,[c['description'] for c in candidates])
        tokens=self.engine.tokenizer(prompt,truncation=True,max_length=self.engine.max_length,return_tensors='pt').to(self.device)
        with torch.inference_mode():logits=self.engine.model(**tokens).logits[0,:len(keys)]
        probabilities=torch.softmax(logits.float(),-1).cpu().tolist()
        result={'decision':dict(zip(keys,probabilities))}
        self.last_raw=dict(prompt=prompt,candidate_ids=keys,distributions=result,device=self.device,protocol='Original candidate-array evaluator; raw logits softmax; no abstention reordering or conditional renormalization',token_count=tokens['input_ids'].shape[-1],max_length=self.engine.max_length)
        return result

    def distributions(self,state,questions):
        from vendored.verdict.core.primitives import Choice,Option,Score,Level,Noul,INSUFFICIENT_EVIDENCE_ID
        def render(value,fallback=''):
            if value in (None,''):return fallback
            return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
        queries=[]
        for name,q in questions.items():
            if q['type']=='choice':
                # The engine owns the abstention candidate. Do not insert a duplicate.
                options=[Option(id=k,description=render(v,k)) for k,v in q['criteria'].items() if k!=INSUFFICIENT_EVIDENCE_ID]
                queries.append(Choice(id=name,question=render(q['instructions']),options=options))
            elif q['type']=='score':queries.append(Score(id=name,question=render(q['instructions']),levels=[Level(id=str(i),description=render(v),value=float(i)) for i,v in enumerate(q['criteria'])]))
            else:
                prop=render(q['instructions']);crit=q.get('criteria') or {}
                if crit:prop+=f" (true: {crit.get('true') or 'yes'}; false: {crit.get('false') or 'no'})"
                queries.append(Noul(id=name,proposition=prop,semantics='conditional_on_sufficient_evidence_v2'))
        text=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)
        out=self.engine.evaluate(text,queries);result={};raw={}
        for (name,q),r in zip(questions.items(),out.results):
            p=dict(r.probabilities);raw[name]=dict(probabilities=p,is_abstention=r.is_abstention)
            if INSUFFICIENT_EVIDENCE_ID not in labels(q):p.pop(INSUFFICIENT_EVIDENCE_ID,None)
            total=sum(p.values())
            if total<=0:raise ValueError('No substantive Verdict probability mass')
            result[name]={k:v/total for k,v in p.items()}
        self.last_raw=dict(native=raw,distributions=result,device=self.device,dtype='float32',abstention='retained when present in source labels; otherwise conditional renormalization following JevBench adapter')
        return result
