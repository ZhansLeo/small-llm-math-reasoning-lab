"""LangGraph 图：把研究 Agent 架构图实现为 7 个节点 + 条件边。

节点 ↔ 架构图概念：
- planner     → Planner：User Goal → Research Plan
- agent       → ReAct Agent：当前研究步骤的推理-行动循环
- tools       → Tool Call：执行 Search / Read / Summarize
- memorize    → Observation → Memory：把工具观察写入记忆
- plan_check  → Plan still valid?：判断 continue / replan / done
- replan      → Replan：计划失效时基于新发现重写计划
- report      → Report：综合目标、计划与记忆撰写最终报告

路由：
- route_after_agent      → 最后一条 AI 消息带工具调用走 tools，否则走 plan_check
- route_after_plan_check → continue→agent / replan→replan / done→report
"""

from __future__ import annotations

import json
import re
import uuid

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from state import ResearchState

MAX_PLAN_STEPS = 6
MAX_REPLANS = 2
MAX_STEP_MESSAGES = 10
MAX_MEMORY_CHARS = 200
MEMORY_LOOKBACK = 8


class ResearchPlan(BaseModel):
    """Planner / Replan 的输出：研究步骤列表。"""

    steps: list[str] = Field(description="3-6 个具体的研究步骤（问题或子任务）")


class PlanCheckDecision(BaseModel):
    """Plan still valid? 的输出。"""

    decision: str = Field(description="continue（继续执行）/ replan（重规划）/ done（完成，写报告）")
    reason: str = Field(description="决策理由")


def build_graph(llm, tools):
    """组装 7 节点扁平图。llm 为支持工具调用的模型，tools 为 build_tools 的返回值。"""

    llm_with_tools = llm.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    def _ask_json(schema: type[BaseModel], prompt: str) -> BaseModel:
        """让模型输出 JSON 并解析为 pydantic 模型。

        DeepSeek 思考模式不支持强制 tool_choice / response_format，
        故用纯文本 prompt + 正则提取 JSON，代替 with_structured_output。
        """
        raw = str(
            llm.invoke(prompt + "\n只输出一个 JSON 对象，不要输出其他文字，不要用代码块包裹。").content
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"模型未返回 JSON: {raw[:200]}")
        try:
            return schema.model_validate_json(match.group(0))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"JSON 解析失败: {e}\n内容: {raw[:300]}") from e

    # —— 检索（Retrieve）：把记忆格式化拼进上下文，供 agent 使用 ——
    def _fmt_memory(memory: list[str]) -> str:
        if not memory:
            return "（暂无）"
        return "\n".join(f"- {m}" for m in memory[-MEMORY_LOOKBACK:])

    def _fmt_plan(plan: list[dict]) -> str:
        return "\n".join(f"{s['id'] + 1}. {s['question']} [{s['status']}]" for s in plan)

    # —— Planner：User Goal → Research Plan ——
    def planner_node(state: ResearchState) -> dict:
        plan = _ask_json(
            ResearchPlan,
            f"研究目标：{state['user_goal']}\n"
            "请把该研究目标拆解为 3-6 个清晰、可执行、彼此独立的研究步骤"
            "（每个步骤是一个具体的研究问题或子任务）。"
            'JSON 格式：{"steps": ["步骤1", "步骤2", ...]}',
        )
        steps = [
            {"id": i, "question": q, "status": "pending"}
            for i, q in enumerate(plan.steps[:MAX_PLAN_STEPS])
        ]
        return {
            "research_plan": steps,
            "current_step": 0,
            "memory": [],
            "replan_count": 0,
            "plan_check_decision": "",
            "report": "",
            "messages": [
                HumanMessage(
                    content=f"研究步骤 1：{steps[0]['question']}",
                    id=str(uuid.uuid4()),
                )
            ],
        }

    # —— ReAct Agent：结合当前步骤 + 已检索记忆，决定调工具或给结论 ——
    def agent_node(state: ResearchState) -> dict:
        step = state["research_plan"][state["current_step"]]
        system_prompt = (
            "你是科研研究助手，按研究计划逐步搜集信息并给出结论。\n"
            f"研究目标：{state['user_goal']}\n"
            f"研究计划：\n{_fmt_plan(state['research_plan'])}\n"
            f"当前步骤：{step['question']}\n"
            f"已掌握的发现（记忆）：\n{_fmt_memory(state['memory'])}\n"
            "要求：优先调用 search 搜索资料、read_page 深入阅读；"
            "当信息足以回答当前步骤时，直接给出简洁结论，不要再调用工具。"
        )
        history = [SystemMessage(content=system_prompt)] + list(state["messages"])
        # 防死循环：推理轮数超限时不再绑定工具，强制模型给出结论
        model = llm if len(state["messages"]) > MAX_STEP_MESSAGES else llm_with_tools
        resp = model.invoke(history)
        return {"messages": [resp]}

    # —— Tool Call：执行 agent 请求的工具调用，产生 Observation ——
    def tools_node(state: ResearchState) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        for call in last.tool_calls:
            tool = tool_by_name.get(call["name"])
            args = call.get("args") or {}
            if tool is None:
                content = f"未知工具: {call['name']}"
            else:
                try:
                    content = str(tool.invoke(args))
                except Exception as e:  # noqa: BLE001
                    content = f"工具执行失败: {e}"
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call["id"],
                    name=call["name"],
                    id=str(uuid.uuid4()),
                )
            )
        return {"messages": tool_messages}

    # —— Observation → Memory：把本轮工具观察去重、截断后写入记忆 ——
    def memorize_node(state: ResearchState) -> dict:
        memory = list(state.get("memory") or [])
        stored = set(memory)  # 已存条目（含前缀），用于去重
        for m in state["messages"]:
            if not isinstance(m, ToolMessage) or not m.content:
                continue
            content = str(m.content)
            if len(content) > MAX_MEMORY_CHARS:
                content = content[:MAX_MEMORY_CHARS] + "…"
            entry = f"[观察] {content}"
            if entry not in stored:
                stored.add(entry)
                memory.append(entry)
        return {"memory": memory}

    def _plan_check_prompt(state: ResearchState, last_answer: str) -> str:
        step = state["research_plan"][state["current_step"]]
        return (
            "你在评估一个研究任务是否按计划推进。\n"
            f"研究目标：{state['user_goal']}\n"
            f"当前研究计划：\n{_fmt_plan(state['research_plan'])}\n"
            f"刚完成的步骤：{step['question']}\n"
            f"该步骤结论：{last_answer or '（无）'}\n"
            f"已掌握的发现：{_fmt_memory(state['memory'])}\n"
            "请判断：\n"
            "- continue：计划仍有效，继续执行下一个待完成步骤；\n"
            "- replan：新发现表明当前计划需要调整（目标未变，但步骤要改写）；\n"
            "- done：所有步骤已完成且信息足以撰写报告。"
        )

    # —— Plan still valid?：判断 continue / replan / done ——
    def plan_check_node(state: ResearchState) -> dict:
        if state.get("replan_count", 0) >= MAX_REPLANS:
            return {"plan_check_decision": "done"}

        last_answer = state["messages"][-1].content if state["messages"] else ""
        decision = _ask_json(
            PlanCheckDecision,
            _plan_check_prompt(state, last_answer)
            + ' JSON 格式：{"decision": "continue|replan|done", "reason": "..."}',
        ).decision.strip().lower()

        if decision == "replan":
            return {"plan_check_decision": "replan"}

        # continue / done 都要先落库当前步骤结论，并把当前步标记为完成
        plan = [dict(s) for s in state["research_plan"]]
        plan[state["current_step"]]["status"] = "done"
        memory = list(state.get("memory") or [])
        if last_answer:
            memory.append(f"[步骤结论] {last_answer[:MAX_MEMORY_CHARS]}")

        if decision != "continue":
            return {"plan_check_decision": "done", "research_plan": plan, "memory": memory}

        # continue：推进到下一个待完成步骤，清空旧步骤消息、注入新问题
        next_idx = next(
            (i for i, s in enumerate(plan) if s["status"] == "pending"),
            None,
        )
        if next_idx is None:
            return {"plan_check_decision": "done", "research_plan": plan, "memory": memory}
        return {
            "plan_check_decision": "continue",
            "research_plan": plan,
            "memory": memory,
            "current_step": next_idx,
            "messages": [
                *[RemoveMessage(id=m.id) for m in state["messages"] if m.id],
                HumanMessage(
                    content=f"研究步骤 {next_idx + 1}：{plan[next_idx]['question']}",
                    id=str(uuid.uuid4()),
                ),
            ],
        }

    # —— Replan：根据目标与现有发现重写研究计划 ——
    def replan_node(state: ResearchState) -> dict:
        plan = _ask_json(
            ResearchPlan,
            f"研究目标：{state['user_goal']}\n"
            f"已掌握的发现：{_fmt_memory(state['memory'])}\n"
            "原先的研究计划可能已不适用。请基于目标与现有发现，重新拆解为 3-6 个研究步骤。"
            ' JSON 格式：{"steps": ["步骤1", "步骤2", ...]}',
        )
        steps = [
            {"id": i, "question": q, "status": "pending"}
            for i, q in enumerate(plan.steps[:MAX_PLAN_STEPS])
        ]
        return {
            "research_plan": steps,
            "current_step": 0,
            "replan_count": state.get("replan_count", 0) + 1,
            "plan_check_decision": "",
            "messages": [
                *[RemoveMessage(id=m.id) for m in state["messages"] if m.id],
                HumanMessage(
                    content=f"研究步骤 1：{steps[0]['question']}",
                    id=str(uuid.uuid4()),
                ),
            ],
        }

    # —— Report：综合目标、计划与记忆撰写最终报告 ——
    def report_node(state: ResearchState) -> dict:
        prompt = (
            "你是科研报告撰写助手。请基于研究目标、研究计划与已掌握发现，"
            "写一份结构清晰的中文研究报告（Markdown），包含：概述、分主题发现、结论与局限。\n\n"
            f"研究目标：{state['user_goal']}\n"
            f"研究计划：\n{_fmt_plan(state['research_plan'])}\n"
            f"已掌握发现：\n{_fmt_memory(state['memory'])}\n"
        )
        resp = llm.invoke(prompt)
        return {"report": str(resp.content)}

    # —— 路由：agent 之后，有工具调用→tools，否则→plan_check ——
    def route_after_agent(state: ResearchState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "plan_check"

    # —— 路由：plan_check 之后，按决策分流（返回决策键，由条件边 map 翻译成节点）——
    def route_after_plan_check(state: ResearchState) -> str:
        return state.get("plan_check_decision", "done")

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("memorize", memorize_node)
    graph.add_node("plan_check", plan_check_node)
    graph.add_node("replan", replan_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "plan_check": "plan_check"},
    )
    graph.add_edge("tools", "memorize")
    graph.add_edge("memorize", "agent")
    graph.add_conditional_edges(
        "plan_check",
        route_after_plan_check,
        {"continue": "agent", "replan": "replan", "done": "report"},
    )
    graph.add_edge("replan", "agent")
    graph.add_edge("report", END)

    return graph.compile()
