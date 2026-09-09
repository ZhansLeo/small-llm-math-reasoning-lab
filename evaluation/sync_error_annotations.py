"""Synchronize annotations after an evaluator-only rescore, preserving history."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def sync(annotation_path: str) -> dict:
    path = Path(annotation_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = json.loads(Path(payload["source_result"]).read_text(encoding="utf-8"))
    source_by_id = {item["id"]: item for item in source["results"]}
    annotations_by_id = {item["id"]: item for item in payload["errors"]}
    removed = []
    retained = []

    for annotation in payload["errors"]:
        current = source_by_id[annotation["id"]]
        if current["status"] == "correct":
            removed.append({
                "id": annotation["id"],
                "old_status": annotation["status"],
                "new_status": current["status"],
                "new_prediction": current["prediction"],
                "reason": "Removed from model-error set after evaluator-only rescore.",
                "previous_annotation": annotation,
            })
            continue
        annotation["status"] = current["status"]
        annotation["prediction"] = current["prediction"]
        retained.append(annotation)

    added = []
    for current in source["results"]:
        if current["status"] == "correct" or current["id"] in annotations_by_id:
            continue
        item = {
            "id": current["id"], "status": current["status"],
            "question": current["question"], "gold": current["gold"],
            "prediction": current["prediction"], "raw_response": current["raw_response"],
            "primary_error_type": None, "secondary_error_type": None,
            "evidence": "", "notes": "", "review_status": "needs_review",
        }
        retained.append(item)
        added.append(current["id"])

    payload["errors"] = sorted(retained, key=lambda item: item["id"])
    payload.setdefault("evaluator_rescore_history", []).append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": source["evaluator"],
        "removed_now_correct": removed,
        "newly_added_failure_ids": added,
        "remaining_error_count": len(retained),
    })
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    args = parser.parse_args()
    payload = sync(args.annotation_path)
    print(f"Synchronized annotations; remaining errors={len(payload['errors'])}")


if __name__ == "__main__":
    main()
