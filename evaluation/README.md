# GSM8K Evaluation

This directory contains one shared evaluation contract with independent model
backends for local Qwen and the DeepSeek API.

The frozen final findings are in `results/final_holdout_report.md`;
machine-readable integrity, paired, and error analyses are in
`results/final_holdout_audit_i101-300.json`,
`results/final_holdout_comparison_i101-300.json`, and
`results/final_holdout_error_analysis_i101-300.json`.

DeepSeek credentials are loaded from `DEEPSEEK_API_KEY`, an optional
`DEEPSEEK_ENV_FILE`, or the dedicated `(2).env`. The shared `.env` is
intentionally never read or modified by the DeepSeek backend.

## Experiment contract

- Dataset: the first `N` examples of the GSM8K test split, in source order.
- Prompt A: minimal baseline, with no step-by-step instruction.
- Prompt B: step-by-step reasoning.
- Prompt C: structured known-quantities/relationships/target/calculation format.
- Final answer contract: the last line must look like `#### 42`.
- Status: `correct`, `wrong`, or `invalid`. Backend failures stop a run and are
  never counted as model-invalid answers.
- Completed runs contain the exact prompt, decoding configuration, sample
  indices, timestamps, aggregate metrics, and every raw response.

## Commands

Run all fast checks from the project root:

```powershell
python evaluation/test_parser.py
python evaluation/test_prompts.py
python evaluation/test_analysis.py
python evaluation/test_runner.py
```

Prompt A smoke tests:

```powershell
python evaluation/evaluate.py --num-samples 10 --prompt prompt_a --run-label smoke
python evaluation/evaluate_deepseek.py --num-samples 10 --prompt prompt_a --run-label smoke
```

Formal Prompt A runs (DeepSeek first, then Qwen):

```powershell
python evaluation/evaluate_deepseek.py --num-samples 100 --prompt prompt_a
python evaluation/evaluate.py --num-samples 100 --prompt prompt_a
```

Qwen prompt intervention runs:

```powershell
python evaluation/evaluate.py --num-samples 100 --prompt prompt_b
python evaluation/evaluate.py --num-samples 100 --prompt prompt_c
```

Each batch writes a `.checkpoint.json`. Re-running the same command resumes a
matching checkpoint. A mismatched configuration is rejected instead of mixed
into the old run.

## Error annotation

After the Qwen Prompt A baseline finishes:

```powershell
python evaluation/error_analysis.py create evaluation/results/qwen_0.5b_gsm8k_prompt_a_n100.json evaluation/results/qwen_0.5b_gsm8k_prompt_a_n100_errors.json
```

Fill `primary_error_type`, optional `secondary_error_type`, and `evidence` for
every wrong/invalid sample, then validate and summarize:

For a resumable preliminary DeepSeek-assisted pass before human review:

```powershell
python evaluation/classify_errors_deepseek.py evaluation/results/qwen_0.5b_gsm8k_prompt_a_n100_errors.json
```

These labels remain marked `needs_human_review`; they must not be reported as
fully human-validated annotations.

```powershell
python evaluation/error_analysis.py summarize evaluation/results/qwen_0.5b_gsm8k_prompt_a_n100_errors.json evaluation/results/qwen_0.5b_gsm8k_prompt_a_n100_error_summary.json
```
