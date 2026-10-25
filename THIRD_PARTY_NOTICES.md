# Third-party notices

No project license is assigned by this file.

- Clef-Flash and official Clef-Flash GGUF conversion: Apache-2.0 declared,
  https://huggingface.co/Cloudflare/clef-flash and
  https://huggingface.co/ggml-org/Clef-Flash-GGUF. Model revision and file hashes
  are recorded in `docs/downloads/new-decisions-20261015.json`.
- Mica 0.1 model, tokenizer, codebook and author runtime: Apache-2.0,
  https://huggingface.co/sky7350/Mica-v0.1-4B and https://github.com/akivet/Mica.
  Pinned unmodified runtime source, LICENSE and NOTICE are retained under
  `vendored/mica-v01`; our compatibility adapter is separate.
- Pinned llama.cpp ABI headers: MIT. Original bytes, source URLs, hashes and
  upstream LICENSE are retained under `vendored/llama-abi-df18bee`.

- llama.cpp: MIT, <https://github.com/ggml-org/llama.cpp>. Executed through the
  separately built local CUDA image, which includes its upstream license.
- Qwen2.5-0.5B-Instruct model and tokenizer template: Apache-2.0,
  <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>. The local GGUF matches the
  official Qwen GGUF repository's published hash.
- SmolLM2-135M-Instruct model and tokenizer template: Apache-2.0,
  <https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>. The local GGUF
  matches Bartowski's published artifact hash.
- nli-distilroberta-base: Apache-2.0,
  <https://huggingface.co/cross-encoder/nli-distilroberta-base>.
- Kev-0.8B adapter/head and Qwen3.5-0.8B-Base: declared Apache-2.0,
  <https://huggingface.co/jaredpalmer/kev-0.8b> and
  <https://huggingface.co/Qwen/Qwen3.5-0.8B-Base>.
- Laya typed-decisions: declared Apache-2.0,
  <https://huggingface.co/convaiinnovations/laya-typed-decisions>.
- General-purpose openJev Verdict: declared Apache-2.0,
  <https://huggingface.co/heman10x/rlcd-modernbert-151m>.
  Specialized Verdict 2.0 metadata/source references:
  <https://github.com/Heman10x-NGU/openJev-verdict-2.0>. Its abbreviated upstream
  Apache-labeled license text is retained verbatim. The specialized weights
  are unavailable from their published Git LFS server.

Pinned source revisions, download hashes and runtime/license boundaries for
these three decision candidates are described in `docs/decision-models.md` and
`docs/downloads/decision-models.json`. Download-time references remain intact. Reviewed inference/scoring source is now
vendored under `vendored/`, with revisions, local import changes and per-project
licenses recorded in `vendored/manifest.json`. See `specs/benchmarks.md`.

`docs/*tokenizer.json` retains upstream template data and source URLs for
configuration verification. Model weights are local external assets and ignored
by Git. Upstream smoke source links and test names are cited rather than copied.
Apache-2.0 license: <https://www.apache.org/licenses/LICENSE-2.0>.

Additional benchmark sources:

- Public benchmark intake of 2026-10-15: JevBench and SemIf project-authored
  fixtures/code are MIT; MMLU-Pro's GitHub code is Apache-2.0 and its HF
  dataset declares MIT. Kev's wrapper is
  Apache-2.0 and its imported source datasets retain their own terms.
  Exact revisions, unmodified notices, file hashes and row-level origins are
  under `benchmarks/public-20261015/`. Reference-only upstream Python files
  are retained for audit and are not imported by the new trial harness.
  Sources: <https://github.com/fstandhartinger/jevbench>,
  <https://github.com/TheoLeeCJ/SemIf-OpenJev>,
  <https://github.com/jaredpalmer/kev>,
  <https://github.com/TIGER-AI-Lab/MMLU-Pro>,
  <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>.
- DeBERTa-v3-large NLI: declared Apache-2.0,
  <https://huggingface.co/cross-encoder/nli-deberta-v3-large>.
- JevBench original authored tests/scoring: MIT,
  <https://github.com/fstandhartinger/jevbench>.
- Google IFEval original prompts and validators: Apache-2.0,
  <https://github.com/google-research/google-research/tree/master/instruction_following_eval>.
- Typed Decisions: declared Apache-2.0,
  <https://huggingface.co/datasets/LocalLLaMA/typed-decisions>.
- SNLI: declared CC BY-SA 4.0,
  <https://huggingface.co/datasets/stanfordnlp/snli>.
- MNLI, ARC and GSM8K: upstream cards/source terms are preserved under
  `benchmarks/upstream/`; do not replace original dataset attribution with a
  model or wrapper license. Chinese community Laya evaluation declares CC BY 4.0
  with Amazon MASSIVE and synthetic source attribution in its original file.
- NLTK Punkt parameter tables: source snapshot and license notices are retained
  with the original archive at `benchmarks/upstream/nltk-data/`.

Original benchmark assets, identity verification and adaptation boundaries are
described in `docs/benchmark-inventory.md`. No new project license is assigned.
