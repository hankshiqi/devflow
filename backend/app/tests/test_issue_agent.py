# 学习导读｜回归测试：覆盖 issue / agent。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import pytest

from app.services.agents.issue_agent import IssueAgent


class NoopLLM:
    # 测试替身：返回预设结构化结果，保证评测 runner 不会意外访问真实模型。
    async def chat_json(self, system_prompt, user_payload):
        return {}


class InventedOwnerLLM:
    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def chat_json(self, system_prompt, user_payload):
        return {"suggested_owner": "布偶猫", "owner_reason": "仓库文档里提到了布偶猫。"}


# 无模型结果时，规则兜底应能识别明确 Bug 信号。
@pytest.mark.asyncio
async def test_issue_agent_fallback_classifies_bug() -> None:
    result = await IssueAgent(llm=NoopLLM()).run({"title": "API returns 500", "body": "Login fails", "labels": ["bug"]}, {})
    assert result["category"] == "Bug"
    assert result["conclusion"] == "等待澄清"
    assert result["priority"] == "P1"
    assert result["checklist"]


# 强重复证据应形成合并/重复处理建议。
@pytest.mark.asyncio
async def test_issue_agent_recommends_merge_for_strong_duplicate() -> None:
    result = await IssueAgent(llm=NoopLLM()).run(
        {
            "title": "Login API returns 500 when token expires",
            "body": "Steps: expire JWT, call /login, expected 401 but actual 500. Impact: users cannot renew sessions.",
            "labels": ["bug"],
        },
        {"duplicate_candidates": [{"issue_id": "issue-1", "title": "JWT expiry returns 500", "score": 0.91}]},
    )

    assert result["conclusion"] == "合并到已有 Issue"
    assert result["duplicate_candidates"][0]["score"] == 0.91
    assert any(item["source_type"] == "similar_issue" for item in result["evidence"])


# 同一证据重复出现时不应重复计入分析依据。
@pytest.mark.asyncio
async def test_issue_agent_dedupes_repeated_evidence() -> None:
    evidence = {
        "source_type": "workspace_file",
        "title": "backend/app/auth/middleware.py#refresh",
        "snippet": "Refresh expired tokens before validating the session.",
        "source_id": "code-1",
        "score": 0.8,
    }
    result = await IssueAgent(llm=NoopLLM()).run(
        {
            "title": "Expired token refresh fails",
            "body": "Steps: expire JWT, call API, expected refresh but actual 401. Impact: users lose sessions.",
            "labels": ["bug"],
        },
        {"code_references": [evidence], "conversation_evidence": [evidence]},
    )

    matching = [
        item
        for item in result["evidence"]
        if item["source_type"] == "workspace_file" and item["reference"] == "code-1"
    ]
    assert len(matching) == 1


# 桌面/Electron 问题应按真实团队领域信息推荐负责人。
@pytest.mark.asyncio
async def test_issue_agent_matches_desktop_issue_to_electron_owner() -> None:
    result = await IssueAgent(llm=NoopLLM()).run(
        {
            "title": "Feature: 增加桌面化能力",
            "body": "### What problem does this solve?\n可以把这个项目桌面化\n\n### Proposed solution\n解决需要打开网页才能使用的问题",
            "labels": [],
        },
        {
            "team_members": [
                {
                    "name": "李勇宏",
                    "role": "前端工程师",
                    "strengths": "擅长electron、js",
                    "techStack": "js, python",
                }
            ]
        },
    )

    assert result["category"] == "Feature"
    assert result["suggested_owner"] == "李勇宏"
    assert "桌面化/electron" in result["owner_reason"]


# 没有团队候选时不能虚构具体负责人。
@pytest.mark.asyncio
async def test_issue_agent_does_not_assign_person_when_team_is_empty() -> None:
    result = await IssueAgent(llm=NoopLLM()).run(
        {
            "title": "Feature: 增加桌面化能力",
            "body": "希望把项目桌面化，支持 Electron 和窗口管理。",
            "labels": [],
        },
        {"project_documents": [{"title": "README", "snippet": "项目文档里提到了布偶猫 / opus 这样的协作角色。"}]},
    )

    assert result["category"] == "Feature"
    assert result["suggested_owner"] == "待分配"
    assert "还没有添加团队成员" in result["owner_reason"]


# 模型返回的团队外负责人必须被拒绝或修正。
@pytest.mark.asyncio
async def test_issue_agent_rejects_llm_owner_not_in_team_members() -> None:
    result = await IssueAgent(llm=InventedOwnerLLM()).run(
        {
            "title": "Feature: 增加桌面化能力",
            "body": "希望把项目桌面化，支持 Electron 和窗口管理。",
            "labels": [],
        },
        {"team_members": []},
    )

    assert result["suggested_owner"] == "待分配"
    assert "布偶猫" not in result["owner_reason"]
