"""Apply a reviewable correction set to preliminary error annotations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def apply_corrections(annotation_path, corrections_path):
    annotation_path = Path(annotation_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    corrections = json.loads(Path(corrections_path).read_text(encoding="utf-8"))
    allowed = set(payload["taxonomy"])
    by_id = {item["id"]: item for item in payload["errors"]}
    history = []

    for correction in corrections["corrections"]:
        item = by_id[correction["id"]]
        primary = correction["primary_error_type"]
        secondary = correction.get("secondary_error_type")
        if primary not in allowed or (secondary is not None and secondary not in allowed):
            raise ValueError(f"Invalid taxonomy label for id={item['id']}")
        history.append({
            "id": item["id"],
            "previous_primary": item["primary_error_type"],
            "previous_secondary": item.get("secondary_error_type"),
            "previous_evidence": item.get("evidence"),
            "new_primary": primary,
            "new_secondary": secondary,
            "new_evidence": correction.get("evidence", item.get("evidence")),
            "reason": correction["reason"],
        })
        item["primary_error_type"] = primary
        item["secondary_error_type"] = secondary
        if "evidence" in correction:
            item["evidence"] = correction["evidence"]
        if "notes" in correction:
            item["notes"] = correction["notes"]
        item["review_status"] = "assistant_audited_needs_human_review"

    payload.setdefault("audit_history", []).append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": corrections["method"],
        "human_review_required": True,
        "changes": history,
    })
    annotation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path")
    parser.add_argument("corrections_path")
    args = parser.parse_args()
    count = apply_corrections(args.annotation_path, args.corrections_path)
    print(f"Applied {count} audited corrections.")


if __name__ == "__main__":
    main()
