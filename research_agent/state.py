"""Research Agent 的图状态（State）定义。

对应 Agent 架构中的“流动数据”：用户目标、研究计划、当前步骤、记忆、ReAct 消息、最终报告。
"""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class ResearchStep(TypedDict):
    """研究计划中的单个步骤。"""

    id: int
    question: str
    status: Literal["pending", "done"]


class ResearchState(TypedDict):
    # User Goal：用户最初的研究目标
    user_goal: str

    # Research Plan：Planner 产出的研究步骤列表
    research_plan: list[ResearchStep]

    # current_step：当前正在执行的步骤下标
    current_step: int

    # Memory：已积累的观察与步骤结论（Observation → Memory）
    memory: list[str]

    # ReAct Agent 的消息环：当前步骤的推理轨迹（Human/AI/Tool 消息）
    messages: Annotated[list, add_messages]

    # Plan still valid? 最近一次决策：continue / replan / done
    plan_check_decision: str

    # 已重规划次数（防死循环）
    replan_count: int

    # Report：最终研究报告
    report: str
