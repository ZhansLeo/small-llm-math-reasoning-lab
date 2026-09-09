"""Generate or verify the reproducible v1 experiment release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()

LOCAL_MODEL_ARTIFACTS = (
    Path("training/outputs/random_lora_seed42_n1000/adapter/adapter_model.safetensors"),
    Path("training/outputs/error_guided_lora_seed42_n1000/adapter/adapter_model.safetensors"),
)


def content_bytes(path: Path, canonical_text: bool = True) -> bytes:
    data = path.read_bytes()
    if canonical_text:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return data


def sha256(path: Path, canonical_text: bool = True) -> str:
    digest = hashlib.sha256()
    digest.update(content_bytes(path, canonical_text))
    return digest.hexdigest()


def is_release_file(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    name = relative.name
    if relative == Path("RELEASE_MANIFEST.json"):
        return False
    if name in {".env", " (2).env", "(2).env"} or (name.endswith(".env") and name != ".env.example"):
        return False
    if any(part in {".git", ".vscode", ".claude", "__pycache__", ".pytest_cache"} for part in parts):
        return False
    if name.endswith((".pyc", ".pyo", ".checkpoint.json", ".safetensors", ".bin")):
        return False
    if "training" in parts and "outputs" in parts:
        if "checkpoints" in parts or "adapter" in parts or any(part.endswith("_smoke") for part in parts):
            return False
    return True


def release_files() -> list[Path]:
    return sorted(
        (path for path in ROOT.rglob("*") if path.is_file() and is_release_file(path)),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def generate() -> dict:
    tracked = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(content_bytes(path)),
            "sha256": sha256(path),
        }
        for path in release_files()
    ]
    local_artifacts = []
    for relative in LOCAL_MODEL_ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing local frozen adapter: {relative}")
        local_artifacts.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path, canonical_text=False),
            "git_distribution": "excluded_generated_weight",
        })

    audit = json.loads(
        (ROOT / "evaluation/results/final_holdout_audit_i101-300.json").read_text(encoding="utf-8")
    )
    if not audit.get("audit_passed"):
        raise RuntimeError("Cannot freeze a release whose final audit did not pass")

    payload = {
        "schema_version": "1.0",
        "release": f"v{VERSION}",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "dataset": "GSM8K",
            "final_holdout_ids": list(range(101, 301)),
            "evaluator": "v4",
            "decoding": "greedy",
            "best_tested_condition": "base + prompt_a",
            "best_tested_accuracy": 0.47,
        },
        "policy": {
            "holdout": "Do not tune and retest on GSM8K test IDs 101-300.",
            "secrets": "No .env file or API key is part of this release.",
            "weights": "Generated adapter weights are hashed but excluded from ordinary Git distribution.",
        },
        "tracked_files": tracked,
        "local_model_artifacts": local_artifacts,
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def verify(require_local_model_artifacts: bool = False) -> tuple[bool, list[str]]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    issues: list[str] = []
    for item in payload["tracked_files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            issues.append(f"missing: {item['path']}")
        elif len(content_bytes(path)) != item["bytes"] or sha256(path) != item["sha256"]:
            issues.append(f"changed: {item['path']}")

    for item in payload["local_model_artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file():
            if require_local_model_artifacts:
                issues.append(f"missing local model artifact: {item['path']}")
            continue
        if path.stat().st_size != item["bytes"] or sha256(path, canonical_text=False) != item["sha256"]:
            issues.append(f"changed local model artifact: {item['path']}")

    forbidden = [
        item["path"] for item in payload["tracked_files"]
        if Path(item["path"]).name.endswith(".env") and Path(item["path"]).name != ".env.example"
    ]
    if forbidden:
        issues.append(f"secret-like files included: {forbidden}")
    return not issues, issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-local-model-artifacts", action="store_true")
    args = parser.parse_args()
    if args.generate:
        payload = generate()
        print(f"Generated {MANIFEST.name} with {len(payload['tracked_files'])} tracked files.")
    if args.verify or not args.generate:
        ok, issues = verify(args.require_local_model_artifacts)
        print(f"release_verified={ok}")
        for issue in issues:
            print(issue)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
