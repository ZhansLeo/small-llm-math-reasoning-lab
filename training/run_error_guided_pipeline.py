"""Run the resumable Error-guided experiment through the frozen holdout matrix."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
EVALUATION = ROOT / "evaluation"
RESULTS = EVALUATION / "results"
ERROR_ADAPTER = TRAINING / "outputs" / "error_guided_lora_seed42_n1000" / "adapter"


def _run(script, *args):
    command = [sys.executable, str(script), *map(str, args)]
    print("\nPIPELINE:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _completed_result(path, expected_ids):
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("run_status") == "completed"
        and payload.get("dataset", {}).get("dataset_indices") == expected_ids
    )


def _latest_training_checkpoint():
    checkpoint_root = TRAINING / "outputs" / "error_guided_lora_seed42_n1000" / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    checkpoints = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if path.is_dir() and (path / "trainer_state.json").exists():
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((step, path))
    return max(checkpoints, default=(None, None))[1]


def _evaluate_error_guided(count, prompt, start=1, label=None):
    range_label = "" if start == 1 else f"_i{start}-{start + count - 1}"
    label_text = f"_{label}" if label else ""
    output = RESULTS / (
        f"qwen_0.5b_error_guided_sft_gsm8k_{prompt}_n{count}"
        f"{range_label}{label_text}.json"
    )
    expected = list(range(start, start + count))
    if _completed_result(output, expected):
        print(f"PIPELINE: skipping completed {output.name}", flush=True)
        return
    args = [
        "--adapter-path", ERROR_ADAPTER,
        "--condition-slug", "error_guided_sft",
        "--num-samples", count,
        "--start-index", start,
        "--prompt", prompt,
    ]
    if label:
        args.extend(["--run-label", label])
    _run(EVALUATION / "evaluate_lora.py", *args)


def main():
    _run(TRAINING / "mine_base_failures.py")
    _run(TRAINING / "prepare_error_guided_sft.py")
    _run(TRAINING / "test_error_guided.py")

    smoke_adapter = TRAINING / "outputs" / "error_guided_lora_smoke" / "adapter" / "adapter_model.safetensors"
    if not smoke_adapter.exists():
        _run(TRAINING / "train_error_guided_lora.py", "--smoke")
    smoke_result = RESULTS / "qwen_0.5b_error_guided_sft_gsm8k_prompt_a_n10_smoke.json"
    if not _completed_result(smoke_result, list(range(1, 11))):
        _run(
            EVALUATION / "evaluate_lora.py",
            "--adapter-path", smoke_adapter.parent,
            "--condition-slug", "error_guided_sft",
            "--num-samples", "10", "--prompt", "prompt_a", "--run-label", "smoke",
        )

    formal_adapter = ERROR_ADAPTER / "adapter_model.safetensors"
    if not formal_adapter.exists():
        checkpoint = _latest_training_checkpoint()
        if checkpoint is None:
            _run(TRAINING / "train_error_guided_lora.py")
        else:
            print(f"PIPELINE: resuming formal training from {checkpoint}", flush=True)
            _run(
                TRAINING / "train_error_guided_lora.py",
                "--resume-from-checkpoint", checkpoint,
            )
    _evaluate_error_guided(100, "prompt_a")
    _evaluate_error_guided(100, "prompt_c")

    dev_files = [
        RESULTS / "qwen_0.5b_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_random_sft_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_error_guided_sft_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_gsm8k_prompt_c_n100.json",
        RESULTS / "qwen_0.5b_random_sft_gsm8k_prompt_c_n100.json",
        RESULTS / "qwen_0.5b_error_guided_sft_gsm8k_prompt_c_n100.json",
    ]
    _run(
        EVALUATION / "compare_results.py", *dev_files,
        "--output", RESULTS / "error_guided_dev_comparison_n100.json",
    )
    _run(EVALUATION / "freeze_final_protocol.py")
    _run(EVALUATION / "run_holdout_matrix.py")
    print("\nPIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
