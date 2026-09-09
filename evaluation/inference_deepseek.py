"""
DeepSeek API Inference（与本地 Qwen inference.py 独立封装）。

接口对齐：
    from inference import ask_batch           # Qwen 本地
    from inference_deepseek import ask_batch  # DeepSeek API

两者都返回：
    responses = [response1, response2, ...]

不同点：
    - 不在本地加载模型，而是调用 DeepSeek 的 OpenAI 兼容 HTTP API。
    - “batch” 用并发请求实现（线程池），不是 tensor batch。
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from prompts import build_messages


# ============================================================
# Config
# ============================================================

BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

DEFAULT_MAX_NEW_TOKENS = 1024
CONCURRENCY = 8


# ============================================================
# API key
#
# 优先读环境变量 DEEPSEEK_API_KEY；否则只读 DeepSeek 专用 env 文件。
# 原 .env 被其他程序使用，本模块明确不会读取或修改它。
# ============================================================

def _read_key_file(env_path):
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "DEEPSEEK_API_KEY":
            key = value.strip().strip('"').strip("'")
            if key:
                return key
    return None


def _load_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if key:
        return key

    project_root = Path(__file__).resolve().parent.parent
    configured_path = os.environ.get("DEEPSEEK_ENV_FILE")
    candidates = []
    if configured_path:
        path = Path(configured_path)
        candidates.append(path if path.is_absolute() else project_root / path)

    # 当前文件实际带一个前导空格；同时兼容之后重命名为 `(2).env`。
    candidates.extend([
        project_root / "(2).env",
        project_root / " (2).env",
    ])

    for env_path in candidates:
        if env_path.is_file():
            key = _read_key_file(env_path)
            if key:
                return key

    raise RuntimeError(
        "DEEPSEEK_API_KEY not found. "
        "Set the environment variable, DEEPSEEK_ENV_FILE, or use (2).env. "
        "The shared .env is intentionally ignored."
    )


client = None


def _get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=_load_api_key(),
            base_url=BASE_URL,
            timeout=60,
            max_retries=3,
        )
    return client


# ============================================================
# Prompt 来自 prompts.py，与 Qwen 共用同一注册表。
# ============================================================

def _ask_one(question, max_new_tokens, prompt_name):
    resp = _get_client().chat.completions.create(
        model=MODEL,
        messages=build_messages(question, prompt_name),
        max_tokens=max_new_tokens,
        temperature=0,
        stream=False,
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek returned an empty response")

    return content.strip()


# ============================================================
# Batch（并发请求）
# ============================================================

def ask_batch(
    questions,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    prompt_name="prompt_a",
):
    """
    与 Qwen inference.ask_batch 同接口。

    DeepSeek API 一次只处理一个对话，
    所以“batch” = 并发发送多个请求。
    """

    responses = [None] * len(questions)

    def run(i):
        responses[i] = _ask_one(
            questions[i],
            max_new_tokens,
            prompt_name,
        )

    with ThreadPoolExecutor(
        max_workers=CONCURRENCY
    ) as executor:

        # 按输入顺序执行，保持输出顺序与 questions 一致
        list(executor.map(run, range(len(questions))))

    return responses
