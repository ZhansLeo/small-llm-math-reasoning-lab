"""Command-line entry point for local Qwen GSM8K evaluation."""

import argparse
from pathlib import Path

from evaluation_runner import run_evaluation
from inference import DEVICE, MODEL_NAME, ask_batch


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
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
    label = f"_{args.run_label}" if args.run_label else ""
    range_label = (
        "" if args.start_index == 1
        else f"_i{args.start_index}-{args.start_index + args.num_samples - 1}"
    )
    output_name = (
        f"qwen_0.5b_gsm8k_{args.prompt}_n{args.num_samples}"
        f"{range_label}{label}.json"
    )
    run_evaluation(
        ask_batch=ask_batch,
        model_name=MODEL_NAME,
        backend_name="local_transformers",
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
        },
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
