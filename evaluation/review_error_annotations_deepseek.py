"""Second-pass review of preliminary GSM8K failure annotations.

The reviewer sees the official GSM8K rationale in addition to the question,
model response, and first-pass label.  Updates are resumable and preserve an
audit trail.  Output remains explicitly LLM-assisted and requires human review.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from classify_errors_deepseek import TAXONOMY_GUIDE, _clean_text, _parse_json
from inference_deepseek import MODEL, _get_client


def _load_official_answers(annotation_path: Path) -> dict[int, str]:
    dataset_path = annotation_path.resolve().parent.parent / "data" / "gsm8k" / "test.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    return {index: row["answer"] for index, row in enumerate(rows, start=1)}


def _review_batch(items: list[dict], official_answers: dict[int, str]) -> list[dict]:
    compact = []
    for item in items:
        compact.append({
            "id": item["id"],
            "question": item["question"],
            "official_solution": official_answers[item["id"]],
            "gold": item["gold"],
            "prediction": item["prediction"],
            "status": item["status"],
            "model_response": item["raw_response"],
            "first_pass": {
                "primary_error_type": item["primary_error_type"],
                "secondary_error_type": item.get("secondary_error_type"),
                "evidence": item["evidence"],
            },
        })

    prompt = f"""
You are the second-pass auditor for GSM8K failure annotations.

{TAXONOMY_GUIDE}

Use the official solution as the reference calculation. Carefully verify the
first-pass label and evidence against both the model response and official
solution. Correct any factual, numerical, or causal mistake. Do not merely say
that prediction differs from gold. If the model reached the correct value but
submitted a conflicting final answer, use Generation / Formatting Error. If
status is invalid but the reasoning itself is wrong, label the earliest/root
reasoning error and optionally use Generation / Formatting Error as secondary.

Return JSON only:
{{"reviews": [{{"id": 1, "primary_error_type": "exact label", "secondary_error_type": null, "evidence": "specific corrected evidence", "review_note": "what was confirmed or corrected", "confidence": "high|medium|low"}}]}}

Review every item exactly once:
{json.dumps(compact, ensure_ascii=False)}
""".strip()
    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000,
        stream=False,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty review response")
    return _parse_json(content)["reviews"]


def review(annotation_path: str, batch_size: int = 4) -> dict:
    path = Path(annotation_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    official_answers = _load_official_answers(path)
    allowed = set(payload["taxonomy"])
    pending = [item for item in payload["errors"] if item.get("review_status") != "second_pass_reviewed"]
    by_id = {item["id"]: item for item in payload["errors"]}

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        reviews = _review_batch(batch, official_answers)
        expected = {item["id"] for item in batch}
        returned = {item.get("id") for item in reviews}
        if returned != expected:
            raise RuntimeError(f"Review IDs mismatch: expected {expected}, got {returned}")

        for reviewed in reviews:
            primary = reviewed.get("primary_error_type")
            secondary = reviewed.get("secondary_error_type")
            if primary not in allowed or (secondary is not None and secondary not in allowed):
                raise RuntimeError(f"Invalid taxonomy label for id={reviewed.get('id')}")
            if secondary == primary:
                secondary = None
            confidence = reviewed.get("confidence")
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"
            target = by_id[reviewed["id"]]
            target.setdefault("annotation_audit", []).append({
                "stage": "first_pass",
                "primary_error_type": target.get("primary_error_type"),
                "secondary_error_type": target.get("secondary_error_type"),
                "evidence": target.get("evidence"),
                "confidence": target.get("confidence"),
            })
            target.update({
                "primary_error_type": primary,
                "secondary_error_type": secondary,
                "evidence": _clean_text(reviewed.get("evidence")),
                "review_note": _clean_text(reviewed.get("review_note")),
                "confidence": confidence,
                "review_status": "second_pass_reviewed",
            })

        payload["annotation_method"] = {
            "type": "llm_assisted_two_pass",
            "classifier_model": MODEL,
            "reviewer_model": MODEL,
            "review_reference": "official GSM8K rationale",
            "human_review_required": True,
        }
        payload["last_reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Reviewed {min(start + len(batch), len(pending))}/{len(pending)} pending annotations.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    review(args.annotation_path, args.batch_size)


if __name__ == "__main__":
    main()
