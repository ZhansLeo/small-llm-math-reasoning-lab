"""Research Agent 入口：组装 LLM + Tools + Graph，跑一个示例研究任务。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph import build_graph
from tools import build_tools

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)


def get_llm() -> ChatOpenAI:
    """DeepSeek OpenAI-compatible 模型（模型名可用 DEEPSEEK_MODEL 覆盖）。"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        temperature=0.2,
    )


DEMO_GOAL = "调研大语言模型智能体的记忆机制研究进展"


def _print_node(node: str, update: dict) -> None:
    print(f"\n===== 节点: {node} =====")
    for key, value in update.items():
        if key == "messages":
            for m in value:
                kind = type(m).__name__
                content = (m.content or "")[:180].replace("\n", " ")
                calls = getattr(m, "tool_calls", None)
                if calls:
                    print(f"  [{kind}] 请求工具: {[c['name'] for c in calls]}")
                elif content:
                    print(f"  [{kind}] {content}")
        elif key in ("user_goal", "current_step", "replan_count", "plan_check_decision"):
            print(f"  {key}: {value}")
        elif key == "research_plan":
            print(f"  research_plan: {[s['question'][:30] + '[' + s['status'] + ']' for s in value]}")
        elif key == "memory":
            print(f"  memory({len(value)}条): {[m[:40] for m in value]}")
        elif key == "report":
            print(f"  report: {len(value)} 字符")


def main() -> None:
    llm = get_llm()
    tools = build_tools(llm)
    graph = build_graph(llm, tools)

    print(f"研究目标: {DEMO_GOAL}\n" + "=" * 60)
    final_state = None
    for chunk in graph.stream({"user_goal": DEMO_GOAL}, stream_mode="updates"):
        for node_name, update in chunk.items():
            _print_node(node_name, update)
            if node_name == "report":
                final_state = update

    if final_state and final_state.get("report"):
        print("\n\n########## 最终报告 ##########\n")
        print(final_state["report"])
    else:
        print("\n[警告] 未生成报告")


if __name__ == "__main__":
    main()
