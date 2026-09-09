"""Create resumable, DeepSeek-assisted preliminary error annotations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inference_deepseek import MODEL, _get_client


TAXONOMY_GUIDE = """
Choose exactly one primary label and at most one different secondary label:

1. Hallucinated Premise: adds a fact, quantity, entity, or condition absent from the question.
2. Semantic Understanding Error: misunderstands wording, references, or the meaning of a condition.
3. Relation Modeling Error: identifies quantities but constructs the wrong equation, ratio, percentage, or multi-step relation.
4. Goal Misunderstanding: calculates something plausible but answers a different requested target.
5. Spatial / Directional Reasoning Error: mishandles direction, displacement, return travel, or spatial relations.
6. Arithmetic Error: the mathematical setup is correct, but numerical calculation is wrong.
7. Generation / Formatting Error: no reliable final answer, contradiction, truncation, or unusable formatting.

Use the earliest/root cause as primary. Evidence must state the concrete mistake, not merely say the answer differs.
""".strip()


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def _clean_text(value):
    return value.strip() if isinstance(value, str) else ""


def _classify_batch(items):
    compact = [
        {
            "id": item["id"],
            "question": item["question"],
            "gold": item["gold"],
            "prediction": item["prediction"],
            "status": item["status"],
            "model_response": item["raw_response"],
        }
        for item in items
    ]
    prompt = f"""
You are labeling failure modes in a GSM8K evaluation of a small language model.

{TAXONOMY_GUIDE}

Return JSON only in this shape:
{{"annotations": [{{"id": 1, "primary_error_type": "exact label", "secondary_error_type": null, "evidence": "specific concise evidence", "notes": "optional", "confidence": "high|medium|low"}}]}}

Classify every supplied item exactly once. Items:
{json.dumps(compact, ensure_ascii=False)}
""".strip()

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2500,
        stream=False,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty annotation response")
    return _parse_json(content)["annotations"]


def classify(annotation_path, batch_size=4):
    path = Path(annotation_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(payload["taxonomy"])
    pending = [item for item in payload["errors"] if not item["primary_error_type"]]

    payload["annotation_method"] = {
        "type": "llm_assisted_preliminary",
        "model": MODEL,
        "human_review_required": True,
    }

    by_id = {item["id"]: item for item in payload["errors"]}
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        annotations = _classify_batch(batch)
        expected_ids = {item["id"] for item in batch}
        returned_ids = {item.get("id") for item in annotations}
        if returned_ids != expected_ids:
            raise RuntimeError(
                f"Annotation IDs mismatch: expected {expected_ids}, got {returned_ids}"
            )

        for annotation in annotations:
            primary = annotation.get("primary_error_type")
            secondary = annotation.get("secondary_error_type")
            confidence = annotation.get("confidence")
            if primary not in allowed:
                raise RuntimeError(f"Invalid primary label for id={annotation['id']}")
            if secondary is not None and secondary not in allowed:
                raise RuntimeError(f"Invalid secondary label for id={annotation['id']}")
            if secondary == primary:
                secondary = None
            if confidence not in {"high", "medium", "low"}:
                confidence = "low"

            target = by_id[annotation["id"]]
            target.update({
                "primary_error_type": primary,
                "secondary_error_type": secondary,
                "evidence": _clean_text(annotation.get("evidence")),
                "notes": _clean_text(annotation.get("notes")),
                "confidence": confidence,
                "review_status": "needs_human_review",
            })

        payload["last_annotated_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Annotated {min(start + len(batch), len(pending))}/{len(pending)} pending errors.")

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    classify(args.annotation_path, args.batch_size)


if __name__ == "__main__":
    main()
