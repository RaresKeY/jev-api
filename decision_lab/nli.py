"""NLI zero-shot comparison; does not provide native Jev primitives or calibration."""
import json
import time

BINARY_HYPOTHESES = {
 'continuation':'The reply continues or approves the active task.',
 'compact':'The summary preserves all required constraints.',
 'bash':'The task requires executing shell commands.',
 'filesystem':'The task requires reading or editing files.',
 'browser':'The task requires researching web pages.',
 'gui':'The task requires desktop mouse or keyboard interaction.',
 'destructive':'The proposed action permanently deletes existing data.',
 'external':'The proposed action changes a remote production service.',
 'satisfied':'The evidence shows that all requested work is complete.',
 'aligned':'The proposed click is on the correct visible target for the goal.',
 'guidance':'The screen requires human verification or guidance.',
}


class NLIClient:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(self.device).eval()
        label2id = {name.lower(): index for name,index in self.model.config.label2id.items()}
        self.entailment = label2id['entailment']
        self.contradiction = label2id['contradiction']
        self.last_raw = []
        self._logits([('A person eats pizza.','A person eats food.')])

    def _logits(self,pairs):
        inputs = self.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],return_tensors='pt',padding=True,truncation=True,max_length=512).to(self.device)
        with self.torch.inference_mode():
            return self.model(**inputs).logits

    def evaluate(self,state,questions,profile):
        premise = state if isinstance(state,str) else (json.dumps(state,ensure_ascii=False) if profile['premise']=='json' else '\n'.join(f'{key}: {value}' for key,value in state.items()))
        pairs, offsets = [], []
        for name, question in questions.items():
            start = len(pairs)
            if question['type']=='noul':
                hypotheses = [BINARY_HYPOTHESES.get(name,question['instructions'])]
            elif question['type']=='choice':
                hypotheses = ['This situation fits the description: '+str(value) for value in question['criteria'].values()]
            else:
                hypotheses = ['This situation fits the description: '+str(value) for value in question['criteria']]
            pairs += [(premise,hypothesis) for hypothesis in hypotheses]
            offsets.append((name,question,start,len(pairs)))
        logits = self._logits(pairs)
        torch = self.torch
        answers = {}
        for name,question,start,end in offsets:
            subset = logits[start:end]
            if question['type']=='noul':
                if profile['binary']=='entailment':
                    probability = torch.softmax(subset,dim=-1)[0,self.entailment]
                else:
                    probability = torch.softmax(subset[:,[self.contradiction,self.entailment]],dim=-1)[0,1]
                answers[name] = probability.item()
            else:
                distribution = torch.softmax(subset[:,self.entailment],dim=0)
                if question['type']=='choice':
                    answers[name] = list(question['criteria'])[distribution.argmax().item()]
                else:
                    answers[name] = (distribution * torch.arange(len(distribution),device=self.device)).sum().item()
        self.last_raw = {'pairs':pairs,'logits':logits.detach().cpu().tolist(),'id2label':self.model.config.id2label,'device':self.device,'profile':profile}
        return answers

    def pair_distributions(self,pairs):
        """Native three-way NLI: use config labels, never dataset integer IDs."""
        logits=self._logits(pairs)
        probabilities=self.torch.softmax(logits,dim=-1).detach().cpu().tolist()
        names=[str(self.model.config.id2label[i]).lower() for i in range(logits.shape[-1])]
        self.last_raw={'pairs':pairs,'logits':logits.detach().cpu().tolist(),'labels':names,'device':self.device}
        return [dict(zip(names,row)) for row in probabilities]

    def distributions(self,state,questions):
        """Generic typed-question adaptation, independent of workstation field names."""
        from .specialists import labels
        premise=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)
        result={};pairs=[];offsets=[]
        for name,q in questions.items():
            keys=labels(q)
            descriptions=list(q['criteria'].values()) if q['type']=='choice' else q.get('criteria',[]) if q['type']=='score' else [q['instructions'] if isinstance(q['instructions'],str) else json.dumps(q['instructions'],ensure_ascii=False)]
            if q['type']!='noul':
                descriptions=[(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)) if v not in (None,'') else k for k,v in zip(keys,descriptions)]
                hypotheses=['This situation fits the description: '+v for v in descriptions]
            else:hypotheses=descriptions
            offsets.append((name,q,keys,len(pairs),len(pairs)+len(hypotheses)))
            pairs.extend((premise,h) for h in hypotheses)
        logits=self._logits(pairs)
        for name,q,keys,start,end in offsets:
            if q['type']=='noul':
                p=self.torch.softmax(logits[start,[self.contradiction,self.entailment]],-1).detach().cpu().tolist()
            else:p=self.torch.softmax(logits[start:end,self.entailment],0).detach().cpu().tolist()
            result[name]=dict(zip(keys,p))
        self.last_raw={'pairs':pairs,'logits':logits.detach().cpu().tolist(),'distributions':result,'device':self.device,'origin':'zero-shot NLI adaptation; E/(E+C) binary; entailment softmax choices'}
        return result
