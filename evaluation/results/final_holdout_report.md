# Qwen2.5-0.5B GSM8K Final Holdout Report

## 1. Scope and experimental contract

This report covers the frozen GSM8K test subset with one-based IDs 101–300
(`n=200`). The six conditions use the same questions, ordering, greedy decoding,
and answer evaluator. The two LoRA runs use the same 1,000-example, one-epoch
training configuration; their main experimental difference is random versus
Base-failure-mined training examples.

The holdout was opened only after the matrix, adapters, and configurations were
frozen. Its results must not be used to tune a model and then retest on IDs
101–300.

## 2. Statistical and evaluator audit

The independent audit passed every check:

- all six runs are complete and contain exactly IDs 101–300 in the same order;
- questions and gold answers agree across runs;
- all raw responses are non-empty;
- stored predictions, statuses, format flags, and aggregate metrics reproduce
  from the per-example records;
- all runs use greedy decoding and the expected Prompt A/C templates;
- the frozen adapter and configuration hashes still match.

Error analysis exposed a parser omission in evaluator v3. Clear final sentences
such as `Therefore, ... is 20%.` and `Therefore, Lloyd earned $130 ...` were
previously marked invalid. Evaluator v4 adds only two constrained rules:

1. accept an explicit concluding sentence beginning with
   `Therefore/Thus/Hence/So` when it yields a reliable final number;
2. accept a leading-decimal GSM8K submission such as `#### .0333`.

It still rejects ordinary intermediate calculations, truncated responses, and
conclusions that explicitly disclaim the result. All six saved raw-response
runs were rescored uniformly—no model inference was repeated. Each result keeps
its prior v3 aggregate in `rescore_history`; the original v3 comparison and audit
are archived separately.

## 3. Final metrics (evaluator v4)

| Model condition | Prompt | Correct | Wrong | Invalid | Accuracy | 95% Wilson CI | Strict `####` format |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | A | 94 | 102 | 4 | **47.0%** | 40.2–53.9% | 0% |
| Base | C | 82 | 113 | 5 | 41.0% | 34.4–47.9% | 0% |
| Random SFT | A | 69 | 131 | 0 | 34.5% | 28.3–41.3% | 99% |
| Random SFT | C | 60 | 140 | 0 | 30.0% | 24.1–36.7% | 99% |
| Error-guided SFT | A | 71 | 128 | 1 | 35.5% | 29.2–42.3% | 99% |
| Error-guided SFT | C | 63 | 137 | 0 | 31.5% | 25.5–38.2% | 100% |

The Base model has 0% strict-format compliance because it usually answers with
boxed or natural-language conclusions instead of the requested final
`#### number` line. This does not mean its answers are unparseable: evaluator v4
still extracts reliable alternative final-answer forms.

## 4. Paired comparisons

Each row compares outcomes on the same 200 questions. “Lost” means the left
condition was correct and the right was not; “gained” means the reverse.

| Change (left → right) | Accuracy change | Lost | Gained | Net correct | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Base A → Base C | -6.0 pp | 32 | 20 | -12 | 0.1263 |
| Base A → Random A | **-12.5 pp** | 39 | 14 | -25 | **0.0008** |
| Base A → Error-guided A | **-11.5 pp** | 40 | 17 | -23 | **0.0032** |
| Random A → Error-guided A | +1.0 pp | 17 | 19 | +2 | 0.8679 |
| Random A → Random C | -4.5 pp | 28 | 19 | -9 | 0.2430 |
| Error-guided A → Error-guided C | -4.0 pp | 28 | 20 | -8 | 0.3123 |

Interpretation:

- Both SFT variants are worse than Base under Prompt A. The paired transition
  imbalance is unlikely to be sampling noise under the exact McNemar test.
- Error-guided A is only 1 percentage point above Random A. The nearly symmetric
  17-versus-19 transition and `p=0.8679` provide no evidence that failure-mined
  selection outperforms random selection in this setup.
- Prompt C is nominally worse for all three model conditions, but none of the
  within-model A/C differences is statistically significant at 0.05. The valid
  conclusion is “no demonstrated benefit,” not “Prompt C is universally
  harmful.”
- The p-values are unadjusted. They should be read together with effect sizes
  and transition counts, not as standalone proof.

## 5. Base Prompt A error taxonomy

Evaluator v4 leaves 106 Base A failures. Each has one primary label selected by
the earliest/root-cause policy and concrete evidence tied to the raw response.

| Primary error type | Count | Share of 106 failures | Share of all 200 |
|---|---:|---:|---:|
| Relation Modeling Error | **61** | **57.5%** | **30.5%** |
| Semantic Understanding Error | 21 | 19.8% | 10.5% |
| Arithmetic Error | 11 | 10.4% | 5.5% |
| Goal Misunderstanding | 7 | 6.6% | 3.5% |
| Generation / Formatting Error | 3 | 2.8% | 1.5% |
| Hallucinated Premise | 2 | 1.9% | 1.0% |
| Spatial / Directional Reasoning Error | 1 | 0.9% | 0.5% |

The dominant failure is not raw arithmetic. Most errors occur earlier, when the
model must decide which quantities participate, what each quantity refers to,
and which operations encode the wording.

The labels are preliminary: all 106 received a two-pass DeepSeek-assisted
classification using the official GSM8K rationale; four inconsistent cases
(IDs 167, 207, 243, and 299) were corrected by an additional audit against the
official solution and raw response. Full human review is still required before
presenting this taxonomy as manually adjudicated ground truth.

## 6. What the interventions repaired

Recovery counts below consider only the 106 questions Base A failed.

| Base A primary error | Base failures | Base C correct | Random A correct | Error-guided A correct |
|---|---:|---:|---:|---:|
| Relation modeling | 61 | 12 | 7 | 10 |
| Semantic understanding | 21 | 4 | 3 | 3 |
| Arithmetic | 11 | 3 | 1 | 2 |
| Goal misunderstanding | 7 | 1 | 1 | 1 |
| Generation / formatting | 3 | 0 | 2 | 1 |
| Hallucinated premise | 2 | 0 | 0 | 0 |
| Spatial/directional | 1 | 0 | 0 | 0 |

Error-guided A repairs three more Base relation-modeling failures than Random A
(10 versus 7), but this local advantage is offset by regressions elsewhere. It
therefore does not produce a reliable overall gain.

The clearest SFT improvement is answer formatting:

- strict compliance rises from 0% to 99–100%;
- invalid rate falls from 2.0–2.5% for Base to 0–0.5% for LoRA;
- ID 118 is illustrative: Base A calculates 360 correctly but submits
  `\boxed{42}`; both LoRA variants return the correct 360 under both prompts.

That is genuine learning, but it is format/response-policy learning rather than
evidence of stronger general mathematical reasoning.

## 7. Representative failure evidence

### Relation modeling persists (ID 101)

The third friend rang ten more times than the fourth friend. Base A instead
relates the third friend to the second and obtains 140 rather than 175. None of
the six conditions solves the item; predictions range from 75 to 391. This is a
stable reference-resolution/relation error rather than a multiplication slip.

### Semantic interpretation can be repaired (ID 143)

The problem says Jordan plays two hours “every day.” Base A silently assumes a
five-day week and returns 100 instead of `2 × 7 × 10 = 140`. Base C and both
Error-guided prompts solve it; Random A also solves it, while Random C regresses
to 150. This case shows that intervention effects are item-specific and can
interact with prompt structure.

### Arithmetic failure is highly persistent (ID 172)

Base A correctly computes `80×8=640`, `90×6=540`, and `3×10=30`, but sums them
as 1110 instead of 1210. Every one of the six conditions returns 1110. Neither
prompt structure nor this LoRA training budget changed the erroneous arithmetic
trajectory.

### Hallucinated premise (ID 144)

The question gives one price for each shopping item, but Base A invents item
quantities such as five units of milk and six eggs, then concludes $127.60
instead of $16. All six conditions fail this item. This is one of only two
clear hallucinated-premise cases, so it is salient but not the dominant mode.

### Spatial/directional reasoning (ID 158)

Base A treats a run to the 40-yard line as 60 yards from the start and also
counts “back and forth” incorrectly, predicting 2580 instead of an 80-yard
difference. All six conditions fail, indicating that the training intervention
did not address this rare spatial structure.

### Wrong target (ID 250)

Base A computes Sue’s total cookie calories (9200), but the question asks how
many more calories she consumed than her sister. It never subtracts the
sister’s 18 cookies, so the requested difference of 5600 is not produced. No
condition solves the item.

### True generation failure (ID 293)

Base A falls into a repetitive loop around “3 hours + 30 minutes” until the
generation limit, with no final answer. This remains a genuine invalid after the
parser correction; alternative conditions give explicit but incorrect numeric
answers.

## 8. Evidence of SFT regression

The lower SFT accuracy is not explained by stricter parsing. Random A loses 39
questions that Base A solved and gains only 14; Error-guided A loses 40 and
gains 17.

Representative Base-correct → both-SFT-wrong cases:

- **ID 102:** gold 6; Base A 6, Random A 39, Error-guided A 35.25. Both LoRA
  models lose the “remaining pages divided by four days” relation. Prompt C
  restores 6 for both adapters, showing a prompt–adapter interaction.
- **ID 113:** gold 24; Base A 24, Random A 12, Error-guided A 48. The two LoRA
  variants move in opposite directions, while Prompt C restores 24 for both.
- **ID 121:** gold 240; Base A 240, Random A 104, Error-guided A 41.6. All Prompt
  C variants also fail, so this is a broad post-training regression.
- **ID 130:** gold 10,000; Base A 10,000, Random A 7,125, Error-guided A 600,000.
  This is a large relation/unit regression in the hospital hourly-profit problem.
- **ID 192:** gold 5; Base A 5, both LoRA Prompt A runs return 35 by omitting the
  conversion from weekly to daily consumption. Prompt C restores 5 for both.

These transitions support a catastrophic-forgetting or distribution-shift
hypothesis: the adapters learn the requested answer style strongly but disturb
some previously correct reasoning behavior. This remains a hypothesis, not a
demonstrated mechanism. Plausible contributors include the small 1,000-example
budget, one epoch of relatively concentrated updates, hard-example bias in the
failure-mined set, and lack of an explicit replay/retention mixture containing
Base-correct examples.

## 9. Final conclusions

1. The best tested condition is the unmodified Base model with Prompt A at
   **47.0%** accuracy.
2. Structured Prompt C shows no statistically demonstrated benefit and is
   nominally 4–6 percentage points worse in every model condition.
3. Both LoRA variants successfully learn strict output formatting and nearly
   eliminate invalid answers.
4. Neither LoRA variant improves mathematical accuracy; both significantly
   underperform Base A in paired testing.
5. Failure-mined Error-guided SFT is only +1 point above Random SFT and is not
   statistically distinguishable from it.
6. Relation modeling is the main preliminary Base failure mode (61/106), much
   larger than pure arithmetic error (11/106).
7. The next training experiment should be treated as a new experiment on a new
   holdout. A defensible design would mix hard failure examples with Base-correct
   replay examples and tune on validation—not reuse IDs 101–300.

## 10. Reproducible artifacts

- `final_holdout_audit_i101-300.json`: independent v4 integrity audit and Wilson intervals.
- `final_holdout_comparison_i101-300.json`: six-run paired comparison and exact McNemar tests.
- `final_holdout_error_analysis_i101-300.json`: taxonomy recovery tables, transitions, and representative cases.
- `qwen_0.5b_gsm8k_prompt_a_n200_i101-300_errors.json`: all 106 Base A failure annotations with raw responses, evidence, and review history.
- `final_holdout_comparison_i101-300_evaluator_v3.json`: archived pre-correction comparison.
- `final_holdout_audit_i101-300_evaluator_v3.json`: archived pre-correction audit.
