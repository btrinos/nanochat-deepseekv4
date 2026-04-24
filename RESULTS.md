# Results

These notes summarize the current full-data evaluation work. All committed
headline numbers are seed-averaged across 3 seeds. The medium preset is
implemented but not yet measured, so the current evidence is still a small
configuration smoke test.

## Bugs Found And Fixed

- Aux-free expert-bias balancing was wired into the optimizer step so
  `gate_bias` is updated after each DeepSeek-V4 optimizer step.
- Hash routing was changed from adjacent-token linear offsets to deterministic
  fixed random permutation tables per layer. This avoids adjacent byte IDs
  sharing almost the same expert set.
- The attention sink is now represented by an explicit sink position instead of
  relying on the sink KV entry being appended last.
- The README and results notes now describe the MoE objective as expert-bias
  balancing with a light sequence-wise auxiliary term, not purely
  aux-loss-free training.
- The parameter-matched GPT selector now keeps each attention head dimension
  even, which avoids invalid RoPE dimensions.

## Architecture Changes And Ablations

The small run compares the full DeepSeek-V4 variant against two ablations on
WikiText-2 validation:

| model | validation loss | perplexity | loss delta vs full |
|---|---:|---:|---:|
| full | 3.0236 +/- 0.0127 | 20.57 +/- 0.26 | +0.0000 |
| no hash routing | 3.0223 +/- 0.0069 | 20.54 +/- 0.14 | -0.0013 |
| no shared expert | 3.0454 +/- 0.0045 | 21.02 +/- 0.09 | +0.0218 |

What helped in this small run:

- The full DeepSeek-V4 variant beats the native GPT baseline and the
  active-parameter-matched GPT baseline on both datasets and all final splits.
- The shared expert appears useful at this scale: removing it makes WikiText-2
  validation loss worse by `+0.0218`.

What did not show a clear small-scale benefit:

- Hash routing is tied with the full model in this small run. The no-hash
  ablation is slightly lower by `0.0013`, well inside the scale where this
  should be treated as no evidence of benefit.
- The parameter-matched GPT baseline is worse than the smaller native GPT in
  the small preset. That does not mean parameter matching is unimportant; it
  means the specific 4-layer, width-96 GPT selected for active-parameter
  matching was harder to optimize in the short 100-step budget.

## Final Headline Table

All values are mean +/- sample standard deviation across 3 seeds.

| config | dataset | split | model | loss | perplexity | bits/byte |
|---|---|---|---|---:|---:|---:|
| small | Tiny Shakespeare | validation | native GPT | 3.2734 +/- 0.0090 | 26.40 +/- 0.24 | 4.723 +/- 0.013 |
| small | Tiny Shakespeare | validation | param-matched GPT | 3.4302 +/- 0.0060 | 30.88 +/- 0.19 | 4.949 +/- 0.009 |
| small | Tiny Shakespeare | validation | DeepSeek-V4 variant | 3.1144 +/- 0.0165 | 22.52 +/- 0.37 | 4.493 +/- 0.024 |
| small | Tiny Shakespeare | test | native GPT | 3.2980 +/- 0.0100 | 27.06 +/- 0.27 | 4.758 +/- 0.014 |
| small | Tiny Shakespeare | test | param-matched GPT | 3.4564 +/- 0.0077 | 31.70 +/- 0.24 | 4.987 +/- 0.011 |
| small | Tiny Shakespeare | test | DeepSeek-V4 variant | 3.1303 +/- 0.0178 | 22.88 +/- 0.41 | 4.516 +/- 0.026 |
| small | WikiText-2 | validation | native GPT | 3.1308 +/- 0.0071 | 22.89 +/- 0.16 | 4.517 +/- 0.010 |
| small | WikiText-2 | validation | param-matched GPT | 3.2672 +/- 0.0126 | 26.24 +/- 0.33 | 4.714 +/- 0.018 |
| small | WikiText-2 | validation | DeepSeek-V4 variant | 3.0236 +/- 0.0127 | 20.57 +/- 0.26 | 4.362 +/- 0.018 |
| small | WikiText-2 | test | native GPT | 3.1275 +/- 0.0074 | 22.82 +/- 0.17 | 4.512 +/- 0.011 |
| small | WikiText-2 | test | param-matched GPT | 3.2626 +/- 0.0120 | 26.12 +/- 0.31 | 4.707 +/- 0.017 |
| small | WikiText-2 | test | DeepSeek-V4 variant | 3.0222 +/- 0.0126 | 20.54 +/- 0.26 | 4.360 +/- 0.018 |
| medium | Tiny Shakespeare | validation | native GPT | not measured | not measured | not measured |
| medium | Tiny Shakespeare | validation | param-matched GPT | not measured | not measured | not measured |
| medium | Tiny Shakespeare | validation | DeepSeek-V4 variant | not measured | not measured | not measured |
| medium | Tiny Shakespeare | test | native GPT | not measured | not measured | not measured |
| medium | Tiny Shakespeare | test | param-matched GPT | not measured | not measured | not measured |
| medium | Tiny Shakespeare | test | DeepSeek-V4 variant | not measured | not measured | not measured |
| medium | WikiText-2 | validation | native GPT | not measured | not measured | not measured |
| medium | WikiText-2 | validation | param-matched GPT | not measured | not measured | not measured |
| medium | WikiText-2 | validation | DeepSeek-V4 variant | not measured | not measured | not measured |
| medium | WikiText-2 | test | native GPT | not measured | not measured | not measured |
| medium | WikiText-2 | test | param-matched GPT | not measured | not measured | not measured |
| medium | WikiText-2 | test | DeepSeek-V4 variant | not measured | not measured | not measured |

## Remaining Caveats

- The committed numbers are small-config results: `n_layer=2`, `n_embd=128`,
  `n_head=4`, `seq_len=64`, and 100 optimizer steps.
- `seq_len=64` does not strongly stress compressed attention, long-context
  behavior, or the sliding-window branch.
- The benchmark is byte-level, so early training is dominated by local byte and
  character statistics.
- The DeepSeek-V4 variant includes many interacting mechanisms. The small run
  can identify obvious regressions and promising signals, but it cannot isolate
  every component.
- The medium config needs a CUDA-class run before making any headline
  architecture claim.
