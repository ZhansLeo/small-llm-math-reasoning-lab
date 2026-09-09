# Random LoRA/SFT

This experiment trains a standalone rank-8 LoRA adapter from the original
`Qwen/Qwen2.5-0.5B-Instruct` checkpoint. It never modifies or merges the base
model.

## Fixed data contract

- 1,000 seeded random examples from GSM8K train for SFT.
- 100 separate GSM8K train examples for validation.
- GSM8K test 1–100 is diagnostic/dev.
- GSM8K test 101–300 is sealed for the final Base/Random/Error-guided comparison.
- Assistant targets contain cleaned GSM8K reasoning and end with `#### NUMBER`.
- Only assistant tokens contribute to loss.

## Commands

```powershell
python training/prepare_random_sft.py
python training/test_training_data.py
python training/train_random_lora.py --smoke
python training/train_random_lora.py
```

If a full run is interrupted, resume the latest checkpoint explicitly:

```powershell
python training/train_random_lora.py --resume-from-checkpoint training/outputs/random_lora_seed42_n1000/checkpoints/checkpoint-N
```

If memory is insufficient, rerun with `--gradient-checkpointing` without
changing the statistical experiment configuration.

Evaluate the saved adapter on diagnostic/dev:

```powershell
python evaluation/evaluate_lora.py --adapter-path training/outputs/random_lora_seed42_n1000/adapter --condition-slug random_sft --num-samples 10 --prompt prompt_a --run-label smoke
python evaluation/evaluate_lora.py --adapter-path training/outputs/random_lora_seed42_n1000/adapter --condition-slug random_sft --num-samples 100 --prompt prompt_a
python evaluation/evaluate_lora.py --adapter-path training/outputs/random_lora_seed42_n1000/adapter --condition-slug random_sft --num-samples 100 --prompt prompt_c
```

Do not run test indices 101–300 until the Error-guided SFT condition is fixed
and trained.

## Completed run

The seeded 1,000-example run completed successfully. Its best checkpoint was
step 125 with validation loss 0.5439. On diagnostic/dev, the adapter scored
34% with Prompt A and 31% with Prompt C; strict format compliance was 100% and
99%, respectively. See
`evaluation/results/stage3_random_sft_report.md` for the controlled comparison
and interpretation.

## Error-guided SFT

Mine failures from the unadapted base model, prepare the selected official
solutions, and train with the shared controlled configuration:

```powershell
python training/mine_base_failures.py
python training/prepare_error_guided_sft.py
python training/test_error_guided.py
python training/train_error_guided_lora.py --smoke
python training/train_error_guided_lora.py
```

The mining command checkpoints after every batch and resumes automatically.
Backend failures stop the run and are never selected as model failures.

To run the entire remaining sequence—including protocol freeze and the final
six-condition holdout matrix—use:

```powershell
python training/run_error_guided_pipeline.py
```

## Frozen final outcome

The six-condition holdout on GSM8K test IDs 101–300 is complete. Base Prompt A
scored 47.0%, Random SFT Prompt A scored 34.5%, and Error-guided SFT Prompt A
scored 35.5% under evaluator v4. Both adapters strongly learned the answer
format but did not improve mathematical accuracy. See
`evaluation/results/final_holdout_report.md` for the paired analysis.
