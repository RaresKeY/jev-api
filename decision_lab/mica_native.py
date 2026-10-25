"""Pinned Mica author readout over verified llama.cpp shared libraries, no generation."""
import ctypes
import json
from pathlib import Path
import sys
from .evaluation import ROOT
from .specialists import TypedDistributionClient


class MicaClient(TypedDistributionClient):
    def __init__(self, abi_path, seqs=1, context=4096):
        sys.path.insert(0, str(ROOT / 'vendored/mica-v01'))
        from mica import direct_backend as native
        from mica import typesafe_server as wire
        from transformers import PreTrainedTokenizerFast

        # The pinned runtime adds this field to the author's b11010 ABI.
        # No author inference or prompt code is changed. Reject ABI mismatches.
        fields = list(native.llama_context_params._fields_)
        if not any(name == 'moe_cache_size' for name, _ in fields):
            fields.insert(next(i for i, (name, _) in enumerate(fields) if name == 'abort_callback'),
                          ('moe_cache_size', ctypes.c_size_t))
        class CurrentContext(ctypes.Structure):
            _fields_ = fields
        abi = json.loads(Path(abi_path).read_text())
        actual = dict(model_size=ctypes.sizeof(native.llama_model_params),
                      context_size=ctypes.sizeof(CurrentContext),
                      **{name: getattr(CurrentContext, name).offset
                         for name in ('moe_cache_size', 'abort_callback', 'ctx_other')})
        if actual != abi:
            raise RuntimeError('Mica binding does not match compiled runtime headers')
        native.llama_context_params = CurrentContext
        directory = ROOT / 'models/mica-01-q8'
        # Generic fast loader avoids assuming support for the newer config class
        # in this retained container. Tokenizer bytes and template are unmodified.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(directory / 'tokenizer.json'),
            chat_template=(directory / 'chat_template.jinja').read_text(),
            eos_token='<|im_end|>', pad_token='<|endoftext|>',
            clean_up_tokenization_spaces=False)
        from mica.codebook import verify
        self.codebook = verify(tokenizer)
        self.judge = native.DirectJudge.from_calibration(
            directory / 'mica-v0.1-4b-Q8_0.gguf', tokenizer,
            directory / 'calibration.json', runtime_dir=ROOT / '.local/llama-df18bee-runtime',
            n_ctx=context, n_seq_max=seqs, max_length=context, n_gpu_layers=-1,
            flash_attn=1, n_ubatch=512)
        self.wire = wire
        self.seqs = seqs
        self.abi = actual
        self.last_raw = None

    def distributions(self, state, questions):
        self.last_raw = None
        rows = self.wire.rows_from_request(dict(state=state, questions=questions))
        probabilities = self.judge.judge(rows, share_prefix=self.seqs > 1)
        answers = {row['id']: self.wire.answer(row, values)
                   for row, values in zip(rows, probabilities)}
        from mica.native import prompt_for
        tokens = sum(len(self.judge.tokenize(prompt_for(row, self.judge.tokenizer))) for row in rows)
        self.last_raw = dict(model='mica-v0.1-4b', answers=answers,
                             usage=dict(input_tokens=tokens, output_tokens=0))
        result = {}
        for row, values in zip(rows, probabilities):
            result[row['id']] = dict(zip((item['id'] for item in row['candidates']), values))
        return result

    def close(self):
        self.judge.close()
