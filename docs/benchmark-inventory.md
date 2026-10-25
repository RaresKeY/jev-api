# Benchmark source and readiness inventory

Original bytes are preserved under `benchmarks/upstream/`. Acquisition manifests pin revisions, file identities, hashes and source URLs; `benchmarks/inventory.json` validates labels, IDs and soft targets. The main runner has no cloud backend.

## Model-to-benchmark map

| Downloaded model | Publisher/author benchmark evidence | Deployed original data | Limits on comparison |
|---|---|---|---|
| Kev-0.8B | [Model card](https://huggingface.co/jaredpalmer/kev-0.8b), [author suites](https://github.com/jaredpalmer/kev/tree/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/evals): decision-v7, transfer-v4/v9, documents, hard skills, breadth, tasksource, devtools, long documents/When2Call | decision-v7, transfer-v4 and hard-v1 original test files; full manifests and exact native metric implementation | transfer-v4 has 764 records but only 656 clean records: the remaining 108 are option/none variants. Native metrics retain this filter. Published development scores and locked-test scores are distinct. Breadth manifest points to `jaredpalmer/kev-private-evals`; large private sets are not deployable from the available public mirror. |
| Laya typed-decisions | [Card](https://huggingface.co/convaiinnovations/laya-typed-decisions), [typed dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), [author benchmark docs](https://github.com/NandhaKishorM/laya/blob/fa9a2a7070b1789912a49ae24603bbfb1a78b001/BENCHMARKS.md) | 400 cases/2,000 soft-gold decisions across four workflows; 219 original Chinese community cases/284 decisions | This checkpoint is specialized to the four workflows. Gold is an average of three teacher samples, not human truth. External source evaluation code/logs are sponsor-only; our documented KL/Brier definitions are available, but cannot be asserted byte-identical to that private harness. |
| General openJev Verdict | [Model card](https://huggingface.co/heman10x/rlcd-modernbert-151m), [author evaluator](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/bff28567cff463b833bf044f351a8b7945d53e07/scripts/evaluate.py), [JevBench](https://github.com/fstandhartinger/jevbench) | Author test set, Banking77/CLINC test and option-count/OOS slices; all public JevBench tiers; typed soft-gold dataset | Exact original candidate order, including abstention position, is preserved by the author-array path. General checkpoint is distinct from the unavailable specialized Verdict 2.0 weights. Typed engine includes abstention; on no-abstention JevBench tasks we follow its published conditional renormalization adapter. |
| Small NLI, DistilRoBERTa | [Card](https://huggingface.co/cross-encoder/nli-distilroberta-base), original AllNLI development CSV | Original SNLI test and MNLI matched/mismatched validation; direct three-way pair inference | AllNLI combines SNLI/MNLI. Their development data informed model training/selection; these are native benchmark checks, not independent domain validation. Invalid SNLI labels are excluded. |
| Larger NLI, DeBERTa-v3-large | [Card](https://huggingface.co/cross-encoder/nli-deberta-v3-large), original AllNLI development CSV | Same original pairs and typed schemas as the small NLI | 435,065,347 parameters vs 82,121,221. Weight file 1,740,312,740 bytes; local safetensors/config/tokenizer hashes verified. FP32 CUDA, maximum 512 pair tokens. Last published CSV rows are about 91.7% vs 87.0% on AllNLI dev; these are author measurements, not our sample results. |
| Qwen2.5-0.5B-Instruct Q4_K_M | [Card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [technical report](https://arxiv.org/abs/2412.15115) | Original ARC, GSM8K, IFEval and NLI content; Kev contains original-derived knowledge/reasoning samples | Current GGUF is quantized. Constrained chat choice differs from publisher multiple-choice likelihood scoring. Original task content alone does not reproduce publisher shot counts, decoding, judges or full report tables. |
| SmolLM2-135M-Instruct Q4_K_M | [Publisher card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), pinned SmolLM2 LightEval README/tasks | Same ARC/GSM8K/IFEval originals | Published instruct tests also include HellaSwag, PIQA, MMLU, CommonsenseQA, TriviaQA, Winogrande, OpenBookQA and MT-Bench. Their publisher protocol is inventoried; full datasets/log-likelihood runner and MT-Bench judge are not claimed implemented here. |
| Jev reference | [JevBench](https://github.com/fstandhartinger/jevbench), [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) independent comparisons | Same original public decision tests can be used for future explicit API runs; existing 24/64-case responses are retained | No fresh Jev calls in this task. JevBench has 231 public decisions and 303 private/judge decisions in its 534-task campaign. We cannot reproduce its private leaderboard/composite or claim Jev scores on our new originals. |

## Available original suites

| Suite | Cases | Questions | Metric/protocol | Full local run |
|---|---:|---:|---|---|
| `jevbench-easy` | 48 | 48 | jevbench-exact | Ready; backend capacity failures recorded |
| `jevbench-standard` | 72 | 72 | jevbench-exact | Ready; backend capacity failures recorded |
| `jevbench-hard` | 111 | 111 | jevbench-exact | Ready; backend capacity failures recorded |
| `kev-transfer` | 764 | 764 | hard-label | Ready; backend capacity failures recorded |
| `kev-decision` | 1,176 | 1,440 | hard-label | Ready; backend capacity failures recorded |
| `kev-hard` | 700 | 1,088 | hard-label | Ready; backend capacity failures recorded |
| `typed-decisions` | 400 | 2,000 | soft-gold | Ready; backend capacity failures recorded |
| `laya-zh` | 219 | 284 | hard-label | Ready; backend capacity failures recorded |
| `verdict-test` | 500 | 500 | hard-label | Ready; backend capacity failures recorded |
| `verdict-real_banking_test` | 1,000 | 1,000 | hard-label | Ready; backend capacity failures recorded |
| `verdict-slice_cardinality_k3` | 100 | 100 | hard-label | Ready; backend capacity failures recorded |
| `verdict-slice_cardinality_k25` | 100 | 100 | hard-label | Ready; backend capacity failures recorded |
| `verdict-slice_missing_option` | 200 | 200 | hard-label | Ready; backend capacity failures recorded |
| `verdict-slice_distant_oos` | 200 | 200 | hard-label | Ready; backend capacity failures recorded |
| `snli` | 9,824 | 9,824 | native-nli | Ready; backend capacity failures recorded |
| `mnli-matched` | 9,815 | 9,815 | native-nli | Ready; backend capacity failures recorded |
| `mnli-mismatched` | 9,832 | 9,832 | native-nli | Ready; backend capacity failures recorded |
| `arc-easy` | 2,376 | 2,376 | task-equivalent-choice | Ready; backend capacity failures recorded |
| `arc-challenge` | 1,172 | 1,172 | task-equivalent-choice | Ready; backend capacity failures recorded |
| `gsm8k` | 1,319 | 1,319 | original-gsm8k-free-response | Dedicated generation runner |
| `ifeval` | 541 | 834 instructions | Original Google strict/loose prompt and instruction checks | Ready for GGUFs |

SNLI source has 10,000 test rows; 176 lack an agreed label and are excluded, leaving 9,824. MNLI validation has 9,815 matched and 9,832 mismatched rows. The typed dataset preserves full gold distributions; every workflow-specific file is retained as well as its all-workflows aggregate. The historical Verdict synthetic test file and old specialized evaluation report are retained as references, without fabricating missing implicit task definitions or substituting the unavailable checkpoint.

## Fidelity and adaptations

- Original data files remain unmodified. Source hashes and row indices connect every adapted case to its original. Gold, latent factors, agreement and rationales never enter inference.
- JevBench scoring code is retained verbatim, including strict and rounded distribution validity, lexical ties and exact winning-level ordinal accuracy. Score expectation/MAE is separate from mode accuracy. The local Odysseus 0.75 tolerance is used only in Odysseus tests.
- Kev native metrics are computed with its original NumPy implementation on clean knowable variants; no distribution is invented for a GGUF label. The common cross-model table includes all selected variants and counts failures, so it is separate from native clean metrics.
- Typed Decisions KL is sum gold × log(gold/prediction), clipping prediction at 1e-12; soft Brier is the summed squared distribution error. Hard-label Brier and 10-bin top-label ECE are also saved distinctly. Teacher agreement is not correctness.
- Original Verdict candidate-array inference preserves exact option and abstention order. Its dedicated typed-question engine is a different protocol; use it for JevBench/typed tasks, where conditional renormalization is documented and original abstention mass stays in raw evidence.
- NLI pair tests map class indices by each model’s config. Generic typed hypotheses are a zero-shot adaptation. NLI truncates pairs at 512; Laya uses checkpoint max_len=1024/head_max_len=256; Verdict uses 512; llama.cpp uses 4096. Context differences are part of the tested deployment.
- ARC chat labels are task-equivalent, not publisher length-normalized likelihood scores. GSM8K uses five original training examples, seed 42 and #### numeric extraction. IFEval uses original prompts and validators. Q4 quantization, served chat templates, sampling and generation limits remain explicit differences from native publisher runs.
- IFEval generation is capped at 1,024 tokens in the partial run; unfinished text is still checked and truncation is reported. Published SmolLM2 task configs and evaluation README are preserved for comparison.

## Licenses and source availability

JevBench authored tasks/code are MIT. Google IFEval code/data and Typed Decisions declare Apache-2.0; Chinese community evaluation states CC BY 4.0 with MASSIVE attribution. SNLI declares CC BY-SA 4.0; MNLI OANC/BNC source terms and ARC/GSM8K license declarations are preserved in dataset cards and metadata. Kev wrappers/code and model weights declare Apache-2.0, but imported underlying datasets retain their own source terms; row-level source repositories/revisions and author manifests are retained, not overwritten with the wrapper license. Verdict imports Banking77/CLINC source terms. Preserve these notices and upstream cards when packaging data. This is not a new project license.

Unavailable portions: JevBench private/judge files; Kev private large suites; specialized Verdict 2.0 checkpoint; sponsor-only Typed Decisions reference harness/logs. Other Smol/Qwen general-language benchmark protocols are inventoried, but the new local runner does not claim a complete lm-eval/LightEval replacement.

## Run commands

Already prepared environment: optional dependencies are pinned in `requirements-benchmarks.txt`; base CUDA Torch/Transformers remain in `requirements-evaluation.txt`. No reinstall or network is needed for current runs. Fresh output directories are mandatory.

```bash
# Restore missing original data at saved revisions, if moving the lab:
.venv/bin/python scripts/restore_benchmarks.py
.venv/bin/python scripts/upstream_benchmarks.py inventory

# Full public easy/standard/hard tiers, no API calls:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/upstream_benchmarks.py run \
  --backends kev laya verdict nli-small nli-large qwen smol \
  --suites jevbench-easy jevbench-standard jevbench-hard \
  --limit 0 --output results/jevbench-full-local-01

# Full typed soft-gold, original Verdict tasks/slices, ARC or native NLI:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/upstream_benchmarks.py run \
  --backends kev laya verdict --suites typed-decisions verdict-test \
  --limit 0 --output results/typed-full-local-01

# Full original generation tests (two separate output directories):
.venv/bin/python scripts/general_benchmarks.py --suite ifeval --limit 0 \
  --max-tokens 2048 --output results/ifeval-full-local-01
.venv/bin/python scripts/general_benchmarks.py --suite gsm8k --limit 0 \
  --shots 5 --max-tokens 1024 --output results/gsm8k-full-local-01

# Derived order pairs stay separate from original benchmark scores:
.venv/bin/python scripts/upstream_benchmarks.py run \
  --case-file benchmarks/derived/option-order-v1.json --limit 0 \
  --output results/order-local-02

# Native-head interactive demo; no separate server/container required:
.venv/bin/python scripts/demo.py --backend kev --interactive
```

Verdict supports at most 25 original candidates (its typed engine admits 24 substantive options plus abstention). A 78-option Kev banking question is expected to be rejected rather than truncated into an easier task. Full runners preserve that rejection in the accuracy denominator. Full 38k-pair runs can take materially longer than the smokes; no full upstream sweep was started.
