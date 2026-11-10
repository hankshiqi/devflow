# 学习导读｜从仓库结构化状态构建推荐提问，过滤已展示项并去重排序。
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.services.agents.base import BaseAgent


class RecommendationAgent(BaseAgent):
    name = "recommendation_agent"
    description = "根据当前仓库的 Issue、PR 和 CI 状态推荐上下文相关的聊天提问。"
    available_tools = ["repository_snapshot"]

    # 从仓库当前结构化数据生成面向用户的推荐问题列表。
    # 参数：input_data：本次任务的业务输入；context：仓库、证据、工具等运行上下文。
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        limit = min(max(int(input_data.get("limit") or 5), 4), 5)
        excluded = {self._normalize(item) for item in input_data.get("exclude") or []}
        candidates = self._build_candidates(input_data)

        selected: list[str] = []
        seen = set(excluded)
        for prompt in candidates:
            normalized = self._normalize(prompt)
            if not normalized or normalized in seen:
                continue
            selected.append(prompt)
            seen.add(normalized)
            if len(selected) >= limit:
                break

        if not selected and excluded:
            for prompt in candidates:
                normalized = self._normalize(prompt)
                if not normalized:
                    continue
                selected.append(prompt)
                if len(selected) >= limit:
                    break

        return {
            "agent": self.name,
            "suggestions": selected,
            "source_counts": {
                "issues": len(input_data.get("issues") or []),
                "pull_requests": len(input_data.get("pull_requests") or []),
                "workflow_runs": len(input_data.get("workflow_runs") or []),
            },
        }

    # 依据 Issue、PR、CI 和会话上下文构造候选提示问题。
    def _build_candidates(self, input_data: dict[str, Any]) -> list[str]:
        issues = sorted(input_data.get("issues") or [], key=self._sort_key, reverse=True)
        prs = sorted(input_data.get("pull_requests") or [], key=self._sort_key, reverse=True)
        runs = sorted(input_data.get("workflow_runs") or [], key=self._sort_key, reverse=True)
        if not issues and not prs and not runs:
            return [
                "概览这个新仓库的当前状态",
                "查看这个项目的代码结构",
                "我该如何同步这个仓库的 Issue、PR 和 CI",
                "帮我建立项目记忆和团队分工",
                "生成初始化项目检查清单",
                "这个仓库接下来应该先配置什么",
                "帮我规划第一次项目同步和分析",
            ]

        open_issues = [item for item in issues if str(item.get("state") or "").lower() == "open"]
        open_prs = [item for item in prs if str(item.get("state") or "").lower() == "open"]
        # 先提取失败 CI，故障修复通常比一般状态查询更值得优先推荐。
        failed_runs = [item for item in runs if str(item.get("conclusion") or "").lower() == "failure"]

        candidates: list[str] = []

        # 按有限数量围绕失败运行生成问题候选，控制推荐规模并突出最近的重要风险。
        for run in failed_runs[:5]:
            name = self._short(run.get("name") or "最近 CI")
            candidates.extend(
                [
                    f"定位 {name} 失败原因",
                    f"把 {name} 的失败日志整理成修复步骤",
                    f"分析 {name} 失败是否和最近 PR 有关",
                    f"给 {name} 生成排查 checklist",
                ]
            )

        if not failed_runs and runs:
            latest = self._short(runs[0].get("name") or "最近 CI")
            candidates.extend(
                [
                    f"检查 {latest} 的 CI 状态",
                    "最近 CI 有没有潜在风险",
                    "总结最近 workflow 的健康状况",
                    "如果 CI 失败，下一步应该先看哪里",
                ]
            )

        # 针对仍开放的 PR 生成审查/合入准备类候选，不把已关闭 PR 混入待办建议。
        for pr in open_prs[:5]:
            number = pr.get("number")
            label = f"PR #{number}" if number else "最新 PR"
            title = self._short(pr.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"总结 {label}{suffix} 的风险",
                    f"检查 {label} 需要补哪些测试",
                    f"给 {label} 生成合并前 checklist",
                    f"{label} 是否会影响 CI 或发布",
                ]
            )

        for issue in open_issues[:5]:
            number = issue.get("number")
            label = f"Issue #{number}" if number else "最新 Issue"
            title = self._short(issue.get("title") or "")
            suffix = f"：{title}" if title else ""
            candidates.extend(
                [
                    f"{label}{suffix} 应该分给谁",
                    f"分析 {label} 的优先级和复杂度",
                    f"给 {label} 写一条澄清评论草稿",
                    f"把 {label} 拆成可执行任务",
                ]
            )

        if failed_runs:
            candidates.extend(
                [
                    "最近 CI 为什么失败",
                    "把失败 CI 和相关 PR 串起来分析" if prs else "把失败 CI 整理成排查步骤",
                ]
            )
        if prs:
            candidates.extend(
                [
                    "总结这个 PR 的风险",
                    "按风险排序当前打开的 PR",
                    "哪些 PR 适合今天合并",
                    "生成一份合并前风险检查清单",
                ]
            )
        if issues:
            candidates.extend(
                [
                    "这个 Issue 应该分给谁",
                    "按优先级整理当前打开的 Issue",
                    "哪些 Issue 需要补充复现信息",
                    "根据团队成员能力推荐 Issue 负责人",
                    "分析打开时间最久的问题",
                ]
            )
        if issues and prs:
            candidates.extend(["把当前 Issue 和 PR 做成站会摘要", "找出最可能拖慢交付的事项"])

        candidates.extend(
            [
                "生成本周研发周报",
                "概览当前仓库状态",
                "找出最值得优先处理的 3 件事",
                "生成一份今日工程状态摘要",
                "梳理当前项目的主要阻塞点",
                "看一下这个项目的代码结构",
                "搜索当前代码里的测试配置",
                "检查最近变更可能带来的风险",
                "帮我制定下一步排查计划",
                "哪些任务需要先找负责人确认",
                "生成一份发给团队的进展说明",
                "检查是否有需要安全确认的写操作",
                "当前仓库有哪些知识库内容可用",
                "本周研发周报应该包含哪些重点",
                "总结当前项目的质量风险",
                "列出今天可以快速推进的事项",
            ]
        )
        # 最后按规范化文本去重并限制结果数量，避免不同规则生成近似重复的问题。
        return self._dedupe(candidates)

    # 规范化后删除重复推荐项，保留首次出现顺序。
    def _dedupe(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = self._normalize(value)
            if not normalized or normalized in seen:
                continue
            result.append(value)
            seen.add(normalized)
        return result

    # 清理推荐文本中的空白及格式差异，便于比较。
    def _normalize(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    # 截断过长展示文本并保留可读边界。
    def _short(self, value: Any, max_length: int = 22) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if len(text) <= max_length:
            return text
        return text[:max_length].rstrip() + "..."

    # 为候选问题生成排序键，优先展示当前最相关的工程事项。
    def _sort_key(self, item: dict[str, Any]) -> float:
        for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
            value = item.get(key)
            if isinstance(value, datetime):
                return value.timestamp()
            if isinstance(value, str) and value:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
        return 0.0
