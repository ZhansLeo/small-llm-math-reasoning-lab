"""Run the frozen six-condition GSM8K 101-300 final matrix sequentially."""

import json
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATION = ROOT / "evaluation"
RESULTS = EVALUATION / "results"
RANDOM_ADAPTER = ROOT / "training" / "outputs" / "random_lora_seed42_n1000" / "adapter"
ERROR_ADAPTER = ROOT / "training" / "outputs" / "error_guided_lora_seed42_n1000" / "adapter"
START = 101
COUNT = 200
END = 300
FREEZE_FILE = RESULTS / "final_protocol_frozen.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_frozen_protocol():
    if not FREEZE_FILE.exists():
        raise FileNotFoundError("Freeze the final protocol before opening holdout")
    frozen = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise RuntimeError("Final protocol is not frozen")
    if frozen.get("holdout", {}).get("indices") != list(range(START, END + 1)):
        raise RuntimeError("Frozen holdout IDs do not match 101-300")
    for artifact in frozen["artifacts"].values():
        path = Path(artifact["path"])
        if not path.exists() or _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Frozen artifact changed or is missing: {path}")


def _completed(path):
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("run_status") == "completed"
        and payload.get("dataset", {}).get("dataset_indices") == list(range(START, END + 1))
    )


def _run(script, args, output):
    if _completed(output):
        print(f"Skipping completed run: {output.name}")
        return
    command = [sys.executable, str(EVALUATION / script), *args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    if not _completed(output):
        raise RuntimeError(f"Run did not produce a valid completed result: {output}")


def main():
    if not RANDOM_ADAPTER.is_dir() or not ERROR_ADAPTER.is_dir():
        raise FileNotFoundError("Both Random and Error-guided adapters must exist before opening holdout")
    _verify_frozen_protocol()

    runs = []
    for prompt in ("prompt_a", "prompt_c"):
        output = RESULTS / f"qwen_0.5b_gsm8k_{prompt}_n200_i101-300.json"
        _run("evaluate.py", ["--num-samples", "200", "--start-index", "101", "--prompt", prompt], output)
        runs.append(output)
    for condition, adapter in (("random_sft", RANDOM_ADAPTER), ("error_guided_sft", ERROR_ADAPTER)):
        for prompt in ("prompt_a", "prompt_c"):
            output = RESULTS / f"qwen_0.5b_{condition}_gsm8k_{prompt}_n200_i101-300.json"
            _run(
                "evaluate_lora.py",
                ["--adapter-path", str(adapter), "--condition-slug", condition,
                 "--num-samples", "200", "--start-index", "101", "--prompt", prompt],
                output,
            )
            runs.append(output)

    comparison = RESULTS / "final_holdout_comparison_i101-300.json"
    subprocess.run(
        [sys.executable, str(EVALUATION / "compare_results.py"),
         *[str(path) for path in runs], "--output", str(comparison)],
        cwd=ROOT, check=True,
    )
    print(f"Final matrix complete: {comparison}")


if __name__ == "__main__":
    main()
