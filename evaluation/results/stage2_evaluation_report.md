# Stage 2 — GSM8K Evaluation Report

## Experimental contract

- Test subset: the first 100 GSM8K test examples in source order.
- Local model: `Qwen/Qwen2.5-0.5B-Instruct`, CPU, batch size 4.
- Reference model: DeepSeek API `deepseek-chat`, concurrency 4.
- Decoding: Qwen greedy (`do_sample=False`); DeepSeek `temperature=0`.
- Evaluator: version `v3`; backend errors are excluded and stop the run.
- Every run uses the same sample IDs, preserves raw responses, and records the
  exact versioned prompt and decoding configuration.

## Main results

| Backend / model | Prompt | Correct | Wrong | Invalid | Accuracy | Invalid rate | Strict format compliance |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek `deepseek-chat` | A | 98 | 2 | 0 | 98% | 0% | 100% |
| Qwen2.5-0.5B-Instruct | A | 41 | 58 | 1 | 41% | 1% | 0% |
| Qwen2.5-0.5B-Instruct | B | 43 | 57 | 0 | 43% | 0% | 0% |
| Qwen2.5-0.5B-Instruct | C | 35 | 64 | 1 | 35% | 1% | 0% |

DeepSeek is a reference backend, not a parameter-matched comparison. Its 98%
result shows that the shared dataset, prompt contract, parser, and scoring path
can support a much stronger model without producing artificial invalid answers.

Qwen frequently ignored the requested final-line format and used `\boxed{}` or
prose instead. Those responses remain valid when the final answer is reliable;
format compliance is therefore reported separately from invalid rate.

## Paired prompt comparison

Because every Qwen run used the same 100 examples, prompt effects were compared
per question rather than from accuracy alone.

| Comparison | Left-only correct | Right-only correct | Net change | McNemar exact p-value |
|---|---:|---:|---:|---:|
| Prompt A vs B | 13 | 15 | B +2 | 0.851 |
| Prompt A vs C | 16 | 10 | C −6 | 0.327 |
| Prompt B vs C | 21 | 13 | C −8 | 0.229 |

At `n=100`, neither Step-by-step nor Structured Reasoning provides statistically
convincing improvement. Prompt B changes which questions succeed, but its net
gain is only two. Prompt C performs worse and gives no evidence that forcing a
five-part structure helps this 0.5B model.

## Preliminary error taxonomy

The following labels are DeepSeek-assisted preliminary annotations. Prompt A
received an additional assistant consistency audit with 13 recorded corrections.
All annotation files retain evidence and remain marked as requiring human review.
Percentages below are counts out of all 100 evaluation questions.

| Primary error type | Prompt A | Prompt B | Prompt C |
|---|---:|---:|---:|
| Hallucinated Premise | 2% | 1% | 1% |
| Semantic Understanding Error | 19% | 11% | 12% |
| Relation Modeling Error | 30% | 34% | 36% |
| Goal Misunderstanding | 3% | 3% | 6% |
| Spatial / Directional Reasoning Error | 1% | 1% | 1% |
| Arithmetic Error | 4% | 7% | 9% |
| Generation / Formatting Error (primary) | 0% | 0% | 0% |

For Prompt A, semantic understanding plus relation modeling accounts for 49 of
the 59 failed questions. Only four failures are primarily arithmetic. The main
limitation is therefore not raw calculation alone: it is translating language
into the correct quantities, relationships, reference sets, and target.

Prompt B reduces the preliminary semantic-error count but increases relation
modeling and arithmetic failures. Prompt C further increases relation, goal, and
arithmetic failures. A plausible interpretation is that longer forced reasoning
creates more opportunities for a small model to introduce an incorrect premise
or equation; this remains a hypothesis until human annotation review is complete.

## Evaluator corrections during validation

The final evaluator supports explicit integers/decimals, comma-separated numbers,
currency, `\boxed{}` answers with units, explicit sentence-final answers with
units or percentages, and explicit boxed fractions. It does not choose among
multiple conflicting submitted answers.

Two initially invalid Prompt A responses were reclassified as wrong after adding
safe support for sentence-final unit answers. One Prompt B response became correct
after supporting `The final answer is \(60\)%`, and one became wrong after adding
explicit boxed-fraction support. Re-scoring used saved raw responses; inference
was not rerun.

## Conclusion and next experiment

The trusted baseline is Qwen Prompt A at 41% accuracy and 1% invalid rate.
Prompt B's 43% is not a reliable improvement on this sample, while Prompt C's
35% suggests that more structure is not automatically better for a 0.5B model.

The next intervention should be LoRA/SFT rather than further prompt elaboration.
Random SFT should establish the training baseline. Error-guided SFT should then
prioritize examples requiring multi-step relation modeling, proportions,
reference-set tracking, and semantic interpretation, while keeping the same 100
test examples untouched for final comparison.
