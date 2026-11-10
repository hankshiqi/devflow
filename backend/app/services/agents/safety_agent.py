# 学习导读｜按动作类型判断风险并生成确认要求，供操作草稿流程使用。
# 分析结果不等于执行授权；真正写入 GitHub 的入口在 action_drafts 路由。
from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    # 分析动作类型与目标内容，返回风险级别和待确认草稿。
    # 参数：input_data：本次任务的业务输入；context：仓库、证据、工具等运行上下文。
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        action = input_data.get("action", "").lower()
        risky = any(
            token in action
            for token in [
                "comment",
                "create_issue",
                "label",
                "close_issue",
                "send_report",
                "write",
                "post",
                "send",
                "评论",
                "创建",
                "新建",
                "发给",
                "发送",
                "改标签",
                "关闭",
                "写回",
            ]
        )
        return SafetyOutput(
            requires_confirmation=risky,
            risk_level="medium" if risky else "low",
            reason="MVP 阶段所有外部写操作只生成草稿。" if risky else "只读分析操作不需要人工确认。",
            draft=input_data.get("draft"),
        ).model_dump()
