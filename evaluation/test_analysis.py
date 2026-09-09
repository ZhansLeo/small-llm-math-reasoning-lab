import json
import tempfile
from pathlib import Path

from compare_results import compare
from error_analysis import TAXONOMY, create_template, summarize


def make_run(path, prompt_name, condition=None):
    payload = {
        "run_status": "completed",
        "model": "fake",
        "backend": "test",
        "prompt": {"name": prompt_name, "version": "v1"},
        "num_samples": 2,
        "correct": 1,
        "wrong": 1,
        "invalid": 0,
        "accuracy": 0.5,
        "invalid_rate": 0.0,
        "format_compliance_rate": 1.0,
        "execution": {} if condition is None else {"condition_slug": condition},
        "results": [
            {
                "id": 1, "question": "q1", "gold": "1",
                "prediction": "1", "status": "correct", "raw_response": "#### 1",
            },
            {
                "id": 2, "question": "q2", "gold": "2",
                "prediction": "3", "status": "wrong", "raw_response": "#### 3",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_tools():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        run_a = root / "a.json"
        run_b = root / "b.json"
        make_run(run_a, "prompt_a")
        make_run(run_b, "prompt_b", "error_guided_sft")

        comparison = compare([run_a, run_b])
        assert comparison["sample_ids"] == [1, 2]
        assert len(comparison["runs"]) == 2
        assert comparison["runs"][0]["condition"] == "base"
        assert comparison["runs"][1]["condition"] == "error_guided_sft"
        assert comparison["pairwise"][0]["right_condition"] == "error_guided_sft"

        annotations_path = root / "annotations.json"
        summary_path = root / "summary.json"
        annotations = create_template(run_a, annotations_path)
        assert len(annotations["errors"]) == 1
        annotations["errors"][0]["primary_error_type"] = TAXONOMY[0]
        annotations["errors"][0]["evidence"] = "Invented an extra quantity."
        annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
        summary = summarize(annotations_path, summary_path)
        assert summary["error_samples"] == 1
        assert summary["primary_error_distribution"][0]["count"] == 1


if __name__ == "__main__":
    test_analysis_tools()
    print("ALL PASSED")
