"""Research Agent 的工具集：Search / Read / Summarize。

对应 Agent 架构图中 Tool Call 的三个子工具：
- search     → 关键词搜索，返回候选资料（本地 Mock 实现，接口可替换为真实搜索）
- read_page  → 读取资料完整正文（Mock 模式从 corpus 取全文；真实 URL 走抓取）
- summarize  → 用 LLM 压缩长文本，供 agent 处理超长正文
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

CORPUS_PATH = Path(__file__).resolve().parent / "mock_corpus.json"
MAX_READ_CHARS = 8000


class SearchBackend(Protocol):
    """搜索后端接口：将来可替换为 DuckDuckGo / Tavily / Bing 等真实实现。"""

    def search(self, query: str) -> list[dict[str, str]]: ...


class MockSearch:
    """在本地 mock_corpus.json 上做关键词匹配的搜索实现。"""

    def __init__(self, corpus_path: Path = CORPUS_PATH) -> None:
        with open(corpus_path, encoding="utf-8") as f:
            self.docs: list[dict[str, Any]] = json.load(f)

    def search(self, query: str) -> list[dict[str, str]]:
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.docs:
            score = self._score(query, doc)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [
            {"title": d["title"], "snippet": d["snippet"], "url": d["url"]}
            for _, d in scored[:5]
        ]

    @staticmethod
    def _score(query: str, doc: dict[str, Any]) -> int:
        haystack = (
            doc["title"] + doc["snippet"] + doc["content"] + " ".join(doc.get("keywords", []))
        ).lower()
        q = query.lower()
        # 按标点/空格切分出的整词片段，命中权重较高
        fragments = [x for x in re.split(r"[\s，。、；：（）()【】,.;:!?]", q) if x]
        score = 2 * sum(1 for f in fragments if f in haystack)
        # 2~3 字滑动窗口，捕捉中文关键词
        grams = {
            q[i : i + n]
            for n in (2, 3)
            for i in range(len(q) - n + 1)
            if not re.search(r"[\s，。、；：（）()【】,.;:!?]", q[i : i + n])
        }
        score += sum(1 for g in grams if g in haystack)
        return score


def build_tools(llm, backend: SearchBackend | None = None) -> list[BaseTool]:
    """构建三个研究工具。llm 用于 summarize；backend 默认用本地 Mock 搜索。"""
    backend = backend or MockSearch()
    docs_by_url = {d["url"]: d for d in getattr(backend, "docs", [])}

    @tool
    def search(query: str) -> str:
        """根据关键词搜索研究资料，返回匹配结果的标题、摘要和 URL（用 read_page 读取完整正文）。"""
        results = backend.search(query)
        if not results:
            return "未找到相关结果，可换用更宽泛的关键词。"
        return "\n\n".join(
            f"[{i + 1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
            for i, r in enumerate(results)
        )

    @tool
    def read_page(url: str) -> str:
        """读取指定 URL 的完整正文，用于深入阅读搜索结果中的某一条资料。"""
        if url in docs_by_url:
            return docs_by_url[url]["content"]
        if url.startswith("mock://"):
            return f"未找到文档: {url}"
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:MAX_READ_CHARS] or "页面无正文"
        except Exception as e:  # noqa: BLE001
            return f"读取失败: {e}"

    @tool
    def summarize(text: str) -> str:
        """用 LLM 把一段长文本压缩成 3-5 条中文要点，每条一行。"""
        resp = llm.invoke(
            "请把下面这段研究资料压缩成 3-5 条中文要点，每条一行，只保留关键信息：\n\n"
            + text[:MAX_READ_CHARS]
        )
        return str(resp.content)

    return [search, read_page, summarize]
