# Results

These notes summarize the current full-data evaluation work. All committed
headline numbers are seed-averaged across 3 seeds. The medium preset is now the
headline result because it uses `seq_len=256`, six layers, and 1,000 optimizer
steps.

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

## Apple Silicon Optimization

The medium run was executed on Apple MPS on an M3 Ultra in float32 with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. The final run took about 53 minutes.

Kept optimizations:

- Cached repeated local and compressed attention masks.
- Used a padded batched MoE expert dispatch for training, where it was faster
  on MPS.
- Kept contiguous per-expert MoE slices for eval, where that path was faster
  than padded batched dispatch.
- Rewrote full non-overlapping split evaluation to use contiguous views instead
  of MPS random gathers.
- Gathered random training byte windows on CPU for MPS runs and moved only the
  small batch to MPS.
- Reused medium `native` results for `param_matched`, because both names resolve
  to the same 6-layer, width-256 GPT and the same seeded initialization.

Tried but not kept:

- BF16/FP16 MPS autocast. It worked in a smoke test, but it was slower than
  float32 for the DeepSeek path on this machine.

## Architecture Changes And Ablations

The small run compares the full DeepSeek-V4 variant against two ablations on
WikiText-2 validation:

| model | validation loss | perplexity | loss delta vs full |
|---|---:|---:|---:|
| full | 3.0236 +/- 0.0127 | 20.57 +/- 0.26 | +0.0000 |
| no hash routing | 3.0223 +/- 0.0069 | 20.54 +/- 0.14 | -0.0013 |
| no shared expert | 3.0454 +/- 0.0045 | 21.02 +/- 0.09 | +0.0218 |

What helped:

- The full DeepSeek-V4 variant beats the native GPT baseline on both datasets
  in the medium run. It also beats the active-parameter-matched GPT baseline;
  for medium, that baseline is identical to native.
- The full DeepSeek-V4 variant also beats both GPT baselines on both datasets
  and all final splits in the small run.
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
| medium | Tiny Shakespeare | validation | native GPT | 1.7250 +/- 0.0305 | 5.61 +/- 0.17 | 2.489 +/- 0.044 |
| medium | Tiny Shakespeare | validation | param-matched GPT | 1.7250 +/- 0.0305 | 5.61 +/- 0.17 | 2.489 +/- 0.044 |
| medium | Tiny Shakespeare | validation | DeepSeek-V4 variant | 1.6066 +/- 0.0218 | 4.99 +/- 0.11 | 2.318 +/- 0.031 |
| medium | Tiny Shakespeare | test | native GPT | 1.8015 +/- 0.0224 | 6.06 +/- 0.14 | 2.599 +/- 0.032 |
| medium | Tiny Shakespeare | test | param-matched GPT | 1.8015 +/- 0.0224 | 6.06 +/- 0.14 | 2.599 +/- 0.032 |
| medium | Tiny Shakespeare | test | DeepSeek-V4 variant | 1.6966 +/- 0.0167 | 5.46 +/- 0.09 | 2.448 +/- 0.024 |
| medium | WikiText-2 | validation | native GPT | 1.7178 +/- 0.0056 | 5.57 +/- 0.03 | 2.478 +/- 0.008 |
| medium | WikiText-2 | validation | param-matched GPT | 1.7178 +/- 0.0056 | 5.57 +/- 0.03 | 2.478 +/- 0.008 |
| medium | WikiText-2 | validation | DeepSeek-V4 variant | 1.6380 +/- 0.0140 | 5.15 +/- 0.07 | 2.363 +/- 0.020 |
| medium | WikiText-2 | test | native GPT | 1.7361 +/- 0.0045 | 5.67 +/- 0.03 | 2.505 +/- 0.006 |
| medium | WikiText-2 | test | param-matched GPT | 1.7361 +/- 0.0045 | 5.67 +/- 0.03 | 2.505 +/- 0.006 |
| medium | WikiText-2 | test | DeepSeek-V4 variant | 1.6571 +/- 0.0098 | 5.24 +/- 0.05 | 2.391 +/- 0.014 |

## Remaining Caveats

- The headline medium numbers are still small in language-model terms:
  `n_layer=6`, `n_embd=256`, `n_head=8`, `seq_len=256`, and 1,000 optimizer
  steps.
- The benchmark is byte-level, so early training is dominated by local byte and
  character statistics.
- The DeepSeek-V4 variant includes many interacting mechanisms. The small run
  can identify obvious regressions and promising signals, but it cannot isolate
  every component.
- The medium run supports a stronger small-scale comparison than the original
  smoke test, but it is still not evidence about production DeepSeek-V4 scale.
