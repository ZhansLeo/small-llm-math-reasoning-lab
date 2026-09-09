# Stage 3 — Random LoRA/SFT Report

## Outcome

The first controlled Random SFT experiment is complete. The adapter learned the
requested answer format very reliably, but it did **not** improve GSM8K answer
accuracy on the 100-example diagnostic development set.

This is a valid negative result. It should be retained as the Random SFT
baseline rather than replaced by an unreported hyperparameter search.

## Experimental contract

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Training source: GSM8K train split.
- Random seed: 42.
- Training examples: 1,000 randomly selected examples.
- Validation examples: 100 disjoint randomly selected examples.
- Training/test exact-question overlap: 0.
- Objective: assistant-only causal-language-model loss.
- Target response: cleaned official GSM8K rationale followed by `#### answer`.
- Sequence limit: 512 tokens; one over-length source example was excluded before
  sampling.
- Epochs: 1.
- Optimizer: AdamW, learning rate `2e-4`, weight decay `0.01`.
- Schedule: linear decay with 4 warmup optimizer steps.
- Micro-batch size: 1; gradient accumulation: 8; effective batch size: 8.
- LoRA: rank 8, alpha 16, dropout 0.05, no bias.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`.
- Trainable parameters: 4,399,104 of 498,431,872 parameters with the adapter
  attached (0.883%).
- Hardware: CPU, float32.
- Selection rule: evaluate every 25 optimizer steps and export the checkpoint
  with the lowest validation loss.

The dataset manifest records the exact source hashes and selected indices. The
training configuration, environment versions, loss history, resource metrics,
and adapter are saved with the training output.

## Training result

| Step | Validation loss |
|---:|---:|
| 25 | 0.5588 |
| 50 | 0.5524 |
| 75 | 0.5469 |
| 100 | 0.5446 |
| 125 | **0.5439** |

- Best checkpoint: step 125.
- Final train loss: 0.5544.
- Training time: 2,509 seconds (41 minutes 49 seconds).
- Peak process RSS: 5.05 GB.
- Exported adapter size: approximately 17.6 MB.

The monotonic validation-loss decrease confirms that optimization was stable.
It does not by itself imply improved exact-answer accuracy: token-level teacher-
forced loss and free-running benchmark accuracy measure different behavior.

## Controlled development-set evaluation

All rows use the same first 100 GSM8K test examples, evaluator v3, greedy
decoding, and maximum 1,024 new tokens. These examples are the existing
prompt-development/diagnostic set, not the sealed final holdout.

| Model | Prompt | Correct | Wrong | Invalid | Accuracy | Strict format |
|---|---:|---:|---:|---:|---:|---:|
| Base | A | 41 | 58 | 1 | 41% | 0% |
| Random SFT | A | 34 | 66 | 0 | 34% | 100% |
| Base | C | 35 | 64 | 1 | 35% | 0% |
| Random SFT | C | 31 | 69 | 0 | 31% | 99% |

Prompt A inference took 253 seconds; Prompt C took 320 seconds on CPU. Every
formal Random SFT result contains 100 non-empty raw responses and a prediction
for every example.

## Paired comparison

| Comparison | Both correct | Base only | Random SFT only | Neither | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Prompt A | 21 | 20 | 13 | 46 | 0.296 |
| Prompt C | 18 | 17 | 13 | 52 | 0.585 |

Random SFT loses a net seven correct answers under Prompt A and four under
Prompt C. Neither difference is statistically compelling at `n=100`, so the
appropriate conclusion is that this configuration provides **no evidence of an
accuracy improvement**. The data do show a large and unambiguous improvement in
strict answer-format compliance.

Within the Random SFT model, Prompt A scores 34% and Prompt C scores 31%.
Structured Reasoning therefore does not recover the lost accuracy in this run.

## Interpretation

The experiment separates two effects that accuracy alone would hide:

1. The adapter successfully learned the supervised response convention.
2. Learning that convention and the 1,000 sampled rationales did not improve
   free-running mathematical reasoning on this diagnostic set.

Plausible contributors include the limited and randomly selected training set,
the mismatch between token-level imitation and exact-answer evaluation, and the
possibility that adapting all attention and MLP projections changes useful base
behavior more than this amount of data can support. These are hypotheses, not
yet established causes.

## Next experiment

Keep this adapter and all results unchanged as the Random SFT condition. The next
primary experiment should be Error-guided SFT using the same base model, LoRA
shape, optimizer settings, sample count, epoch count, evaluator, and prompts.
Only the training-example selection policy should change.

Before constructing that dataset, finish human review of the current error
annotations and define reproducible selection rules for the dominant Semantic
Understanding and Relation Modeling categories. Do not evaluate GSM8K test
indices 101–300 yet. That sealed 200-example holdout should be opened only after
the Error-guided adapter and comparison protocol are frozen, then used once for
the final Base vs Random SFT vs Error-guided SFT comparison.

## Artifacts

- Adapter and training records:
  `training/outputs/random_lora_seed42_n1000/`
- Dataset manifest:
  `training/manifests/random_sft_seed42.json`
- Random SFT Prompt A results:
  `evaluation/results/qwen_0.5b_random_sft_gsm8k_prompt_a_n100.json`
- Random SFT Prompt C results:
  `evaluation/results/qwen_0.5b_random_sft_gsm8k_prompt_c_n100.json`
- Paired comparison:
  `evaluation/results/random_sft_dev_comparison_n100.json`
