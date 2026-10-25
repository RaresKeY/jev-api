# Configuration guidance and observed quirks

The historical chat-runtime profile pins llama.cpp commit `c4a278d` and CUDA 12.6, compiled for
SM75 (RTX 2080 Ti). The [matching server reference](https://github.com/ggml-org/llama.cpp/blob/c4a278d68efa17811006f2123a84081dac03fac7/tools/server/README.md)
is the relevant version, rather than current-main assumptions.

Its [upstream chat tests](https://github.com/ggml-org/llama.cpp/blob/c4a278d68efa17811006f2123a84081dac03fac7/tools/server/tests/unit/test_chat_completion.py)
include template rendering and constrained responses; [tokenizer tests](https://github.com/ggml-org/llama.cpp/blob/c4a278d68efa17811006f2123a84081dac03fac7/tools/server/tests/unit/test_tokenize.py)
cover tokenization. This project adapts those patterns to existing local weights,
without running the complete upstream suite or fetching its TinyLlama fixtures.
The marker smoke requests output that conflicts with the supplied schema. This
checks grammar enforcement rather than coincidental JSON output.

The README advertises top-level `response_format.schema`, but that form was
ignored in live tests of this image. Its upstream tests use
`response_format.json_schema.schema`; this nested form works. The diagnostic
profile intentionally retains the ineffective form to preserve the A/B evidence.

[Qwen's model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) requires its
chat template and declares a 32,768-token context for this checkpoint. Saved
`qwen_generation.json` captures its generation defaults: temperature 0.7,
top-k 20, top-p 0.8, repetition penalty 1.1. Those chat defaults are tested against
neutral greedy classification; they are not assumed optimal for decision gates.
The GGUF reports 630M parameters despite the upstream 0.49B model name. Its SHA256
matches the official Qwen GGUF artifact, so this metadata difference does not
establish weight corruption. A control-looking `</s>` token warning is preserved
in the startup evidence; it is not proof of a broken model.

[SmolLM's model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
provides a sampled example using temperature 0.2 and top-p 0.9. The downloaded
GGUF has an 8,192-token context and a training-run-style general name. It matches
Bartowski's published hash. The card's function-calling statement applies to its
1.7B variant, so it does not establish robust tool classification for 135M.

Official tokenizer templates are saved with source URLs and compared against
`/apply-template` using a system/user sequence with Unicode. Matching templates
and successful constrained output establish configuration compatibility, not
semantic quality. No override is applied without evidence of a mismatch.

The [NLI model card](https://huggingface.co/cross-encoder/nli-distilroberta-base)
identifies label order as contradiction, entailment, neutral. It is trained on
natural-language inference, not shell safety or workstation routing. We compare
raw JSON versus readable premises, entailment softmax versus entailment/contradiction
normalization, and development-selected binary thresholds. It is evaluated
separately through PyTorch; it is not a GGUF chat model.

The [official Jev API reference](https://docs.typesafe.ai/api) defines typed
questions over one state. HTTP binary answers use `noul`; scores are probability-
weighted expectations and can be fractional. The legacy script read the wrong
binary field. A raw-HTTP adapter avoids SDK retry defaults and preserves the
original answer distributions for the one saved reference run.

All sources above are primary sources. Third-party discussion claims about
Qwen template bugs were not used as evidence for an override.
