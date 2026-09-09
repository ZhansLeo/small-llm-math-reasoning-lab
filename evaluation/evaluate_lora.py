"""Evaluate a saved LoRA adapter with the shared GSM8K runner."""

import argparse
from pathlib import Path

from evaluation_runner import run_evaluation
from inference_lora import DEVICE, LoRAInferenceBackend


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--condition-slug", default="random_sft")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--prompt",
        choices=("prompt_a", "prompt_b", "prompt_c"),
        default="prompt_a",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = LoRAInferenceBackend(args.adapter_path)
    label = f"_{args.run_label}" if args.run_label else ""
    range_label = (
        "" if args.start_index == 1
        else f"_i{args.start_index}-{args.start_index + args.num_samples - 1}"
    )
    output_name = (
        f"qwen_0.5b_{args.condition_slug}_gsm8k_{args.prompt}_n{args.num_samples}"
        f"{range_label}{label}.json"
    )
    run_evaluation(
        ask_batch=backend.ask_batch,
        model_name="Qwen/Qwen2.5-0.5B-Instruct+LoRA",
        backend_name="local_transformers_peft",
        data_path=ROOT / "data" / "gsm8k" / "test.jsonl",
        output_path=ROOT / "results" / output_name,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        prompt_name=args.prompt,
        max_new_tokens=args.max_new_tokens,
        decoding={
            "policy": "greedy",
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": args.max_new_tokens,
        },
        execution={
            "device": str(DEVICE),
            "batch_size": args.batch_size,
            "adapter_path": str(Path(args.adapter_path).resolve()),
            "condition_slug": args.condition_slug,
        },
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
