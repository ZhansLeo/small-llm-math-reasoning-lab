# Frozen experiment v1.0.0

Status: **FROZEN**

This release freezes the completed experiment using GSM8K test IDs 101–300.
The final model/prompt matrix, evaluator v4 rescoring, paired statistics, and
error analysis are recorded in `evaluation/results/`.

Rules after this freeze:

1. Do not tune prompts, data selection, LoRA settings, or checkpoints against
   GSM8K test IDs 101–300 and then report new results on the same subset.
2. Corrections to reporting or evaluator logic must retain an audit trail and be
   applied uniformly to every condition from saved raw responses.
3. Any new training intervention must use validation data for model selection
   and a new untouched holdout for its final claim.
4. The Base Prompt A error taxonomy remains model-assisted and must not be
   described as fully human-adjudicated until all 106 labels are reviewed.

The immutable content hashes are listed in `RELEASE_MANIFEST.json` and can be
checked with `python freeze_release.py --verify`.
