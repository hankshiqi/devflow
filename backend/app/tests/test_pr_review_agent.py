# 学习导读｜回归测试：覆盖 pr / review / agent。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import pytest

from app.services.agents.pr_review_agent import PRReviewAgent


class NoopLLM:
    # 测试替身：返回预设结构化结果，保证评测 runner 不会意外访问真实模型。
    async def chat_json(self, system_prompt, user_payload):
        return {}


class OverconfidentLLM:
    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def chat_json(self, system_prompt, user_payload):
        return {
            "merge_recommendation": "建议合入",
            "review_findings": [
                {
                    "severity": "P2",
                    "title": "权限边界缺少验证",
                    "evidence": "diff 修改了权限判断，但没有对应测试证据。",
                    "required_action": "补充权限边界测试或人工验证记录。",
                    "blocking": True,
                }
            ],
            "blocking_issues": [],
        }


# 规则审查发现明确风险时，应产出可追踪的阻塞发现。
@pytest.mark.asyncio
async def test_pr_review_fallback_emits_blocking_findings() -> None:
    result = await PRReviewAgent(llm=NoopLLM()).run(
        {
            "number": 12,
            "state": "open",
            "author": "alice",
            "title": "Update auth middleware",
            "body": "Refactor permission checks.",
            "files": [
                {
                    "filename": "backend/app/auth/middleware.py",
                    "additions": 320,
                    "deletions": 12,
                    "patch": "permission change",
                }
            ],
            "comments": [{"body": "Please verify token handling is not logged."}],
        },
        {"ci_summary": {"failed_runs": 1}},
    )

    severities = {item["severity"] for item in result["review_findings"]}
    assert {"P1", "P2"} <= severities
    assert result["merge_recommendation"] == "暂缓"
    assert result["blocking_issues"]


# 即使模型总体建议合入，具体阻塞发现仍应降低合入建议。
@pytest.mark.asyncio
async def test_pr_review_downgrades_merge_when_llm_returns_blocking_finding() -> None:
    result = await PRReviewAgent(llm=OverconfidentLLM()).run(
        {
            "number": 13,
            "state": "open",
            "author": "alice",
            "title": "Update permission UI",
            "body": "Ready to merge.",
            "files": [
                {
                    "filename": "frontend/app/page.tsx",
                    "additions": 20,
                    "deletions": 3,
                    "patch": "permission UI change",
                }
            ],
            "comments": [],
        },
        {},
    )

    assert result["merge_recommendation"] == "修改后合入"
    assert any("权限边界缺少验证" in item for item in result["blocking_issues"])
    assert result["review_findings"][0]["severity"] == "P2"
