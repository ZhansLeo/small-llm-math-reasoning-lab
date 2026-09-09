import json
import tempfile
from pathlib import Path

from evaluation_runner import run_evaluation


def test_runner():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_path = root / "test.jsonl"
        rows = [
            {"question": "one", "answer": "work\n#### 1"},
            {"question": "two", "answer": "work\n#### 2"},
            {"question": "three", "answer": "work\n#### 3"},
        ]
        data_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        def fake_backend(questions, max_new_tokens, prompt_name):
            assert max_new_tokens == 32
            assert prompt_name == "prompt_a"
            answers = {"one": "#### 1", "two": "#### 9", "three": "no answer"}
            return [answers[q] for q in questions]

        output_path = root / "result.json"
        output = run_evaluation(
            ask_batch=fake_backend,
            model_name="fake",
            backend_name="test",
            data_path=data_path,
            output_path=output_path,
            num_samples=3,
            batch_size=2,
            prompt_name="prompt_a",
            max_new_tokens=32,
            decoding={"policy": "test"},
            execution={"batch_size": 2},
        )
        assert output["correct"] == 1
        assert output["wrong"] == 1
        assert output["invalid"] == 1
        assert output["invalid_rate"] == 1 / 3
        assert output["format_compliance_rate"] == 2 / 3
        assert output["dataset"]["dataset_indices"] == [1, 2, 3]
        assert output["prompt"]["name"] == "prompt_a"
        assert output_path.exists()
        assert not output_path.with_suffix(".checkpoint.json").exists()


def test_backend_failure_is_not_scored_invalid():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_path = root / "test.jsonl"
        data_path.write_text(
            json.dumps({"question": "one", "answer": "#### 1"}) + "\n",
            encoding="utf-8",
        )

        def failed_backend(questions, max_new_tokens, prompt_name):
            raise ConnectionError("simulated API failure")

        output_path = root / "failed.json"
        try:
            run_evaluation(
                ask_batch=failed_backend,
                model_name="fake",
                backend_name="test",
                data_path=data_path,
                output_path=output_path,
                num_samples=1,
                batch_size=1,
                prompt_name="prompt_a",
                max_new_tokens=32,
                decoding={"policy": "test"},
                execution={"batch_size": 1},
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Backend failure should stop the evaluation")

        checkpoint_path = output_path.with_suffix(".checkpoint.json")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["run_status"] == "failed"
        assert checkpoint["num_samples"] == 0
        assert checkpoint["invalid"] == 0
        assert checkpoint["backend_error"]["type"] == "ConnectionError"


def test_resume_from_completed_batch():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        data_path = root / "test.jsonl"
        rows = [
            {"question": "one", "answer": "#### 1"},
            {"question": "two", "answer": "#### 2"},
            {"question": "three", "answer": "#### 3"},
        ]
        data_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        output_path = root / "resume.json"
        calls = 0
        answers = {"one": "1", "two": "2", "three": "3"}

        def interrupted_backend(questions, max_new_tokens, prompt_name):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConnectionError("interrupt after first batch")
            return [f"#### {answers[q]}" for q in questions]

        kwargs = dict(
            model_name="fake",
            backend_name="test",
            data_path=data_path,
            output_path=output_path,
            num_samples=3,
            batch_size=2,
            prompt_name="prompt_a",
            max_new_tokens=32,
            decoding={"policy": "test"},
            execution={"batch_size": 2},
        )
        try:
            run_evaluation(ask_batch=interrupted_backend, **kwargs)
        except RuntimeError:
            pass

        checkpoint_path = output_path.with_suffix(".checkpoint.json")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["num_samples"] == 2

        def resumed_backend(questions, max_new_tokens, prompt_name):
            return [f"#### {answers[q]}" for q in questions]

        output = run_evaluation(ask_batch=resumed_backend, **kwargs)
        assert output["num_samples"] == 3
        assert output["correct"] == 3
        assert not checkpoint_path.exists()


if __name__ == "__main__":
    test_runner()
    test_backend_failure_is_not_scored_invalid()
    test_resume_from_completed_batch()
    print("ALL PASSED")
