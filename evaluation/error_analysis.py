"""Create and summarize evidence-based error annotations for an evaluation run."""

import argparse
import json
from collections import Counter
from pathlib import Path


TAXONOMY = (
    "Hallucinated Premise",
    "Semantic Understanding Error",
    "Relation Modeling Error",
    "Goal Misunderstanding",
    "Spatial / Directional Reasoning Error",
    "Arithmetic Error",
    "Generation / Formatting Error",
)


def create_template(result_path, annotation_path):
    run = json.loads(Path(result_path).read_text(encoding="utf-8"))
    errors = []
    for item in run["results"]:
        if item["status"] == "correct":
            continue
        errors.append({
            "id": item["id"],
            "status": item["status"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": item["prediction"],
            "raw_response": item["raw_response"],
            "primary_error_type": None,
            "secondary_error_type": None,
            "evidence": "",
            "notes": "",
        })

    payload = {
        "source_result": str(Path(result_path).resolve()),
        "taxonomy": list(TAXONOMY),
        "annotation_policy": {
            "primary": "required; choose exactly one taxonomy label",
            "secondary": "optional; choose one different taxonomy label",
            "evidence": "required; cite the specific reasoning failure",
        },
        "errors": errors,
    }
    annotation_path = Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def summarize(annotation_path, summary_path):
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    source_run = json.loads(
        Path(payload["source_result"]).read_text(encoding="utf-8")
    )
    errors = payload["errors"]
    problems = []
    counts = Counter()

    for item in errors:
        primary = item.get("primary_error_type")
        secondary = item.get("secondary_error_type")
        if primary not in TAXONOMY:
            problems.append(f"id={item['id']}: missing/invalid primary_error_type")
        else:
            counts[primary] += 1
        if secondary is not None and secondary not in TAXONOMY:
            problems.append(f"id={item['id']}: invalid secondary_error_type")
        if secondary is not None and secondary == primary:
            problems.append(f"id={item['id']}: secondary duplicates primary")
        if not item.get("evidence", "").strip():
            problems.append(f"id={item['id']}: missing evidence")

    if problems:
        raise ValueError("Annotations are incomplete:\n" + "\n".join(problems))

    total = len(errors)
    summary = {
        "source_annotation": str(Path(annotation_path).resolve()),
        "source_result": payload["source_result"],
        "model": source_run["model"],
        "prompt": source_run["prompt"],
        "evaluation_samples": source_run["num_samples"],
        "annotation_method": payload.get("annotation_method"),
        "human_review_required": True,
        "error_samples": total,
        "primary_error_distribution": [
            {
                "error_type": label,
                "count": counts[label],
                "rate": counts[label] / total if total else 0.0,
                "rate_of_all_samples": (
                    counts[label] / source_run["num_samples"]
                    if source_run["num_samples"] else 0.0
                ),
            }
            for label in TAXONOMY
        ],
    }
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("result_path")
    create.add_argument("annotation_path")

    report = subparsers.add_parser("summarize")
    report.add_argument("annotation_path")
    report.add_argument("summary_path")

    args = parser.parse_args()
    if args.command == "create":
        payload = create_template(args.result_path, args.annotation_path)
        print(f"Created {len(payload['errors'])} error annotations.")
    else:
        summary = summarize(args.annotation_path, args.summary_path)
        print(f"Summarized {summary['error_samples']} error annotations.")


if __name__ == "__main__":
    main()
