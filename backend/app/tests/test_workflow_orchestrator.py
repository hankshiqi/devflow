# 学习导读｜回归测试：覆盖 任务依赖、并发、超时、阻塞、目标一致性与重规划。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.routes.chat import WorkflowTargetResolutionError, _build_conversation_tools, _build_engineering_workflow_runtime
from app.core.config import settings
from app.db.models import AgentTaskRun, AgentWorkflowRun, Conversation, Issue, PRFile, PullRequest, Repository, WorkflowRun
from app.db.session import Base
from app.schemas.workflow import WorkflowClaim, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.skills import get_skill_registry


# 判断合入就绪度需要跨 PR、CI 等领域任务，而不是只看单一结果。
@pytest.mark.asyncio
async def test_planner_builds_cross_domain_workflow_for_merge_readiness() -> None:
    result = await PlannerAgent().run(
        {"message": "这个 PR 能不能合并？请做多 agent 协作 review，检查 Issue、PR 和 CI 风险。"},
        {
            "repo_id": "repo-1",
            "has_issue": True,
            "issue_id": "issue-1",
            "has_pr": True,
            "pr_id": "pr-1",
            "has_failed_ci": True,
            "workflow_run_id": "run-1",
        },
    )

    task_types = [task["task_type"] for task in result["tasks"]]
    assert "repo_health" in task_types
    assert "issue_analysis" in task_types
    assert "pr_review" in task_types
    assert "ci_debug" in task_types
    assert result["acceptance_criteria"]


# 计划确认后仍应分析原来指定对象，不能因出现更新记录而切换目标。
@pytest.mark.asyncio
async def test_confirmed_workflow_keeps_explicit_issue_pr_and_ci_claims_after_newer_records_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)
    captured_pr_input: dict[str, object] = {}

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def capture_pr_run(self, input_data, context):
        captured_pr_input.update(
            {
                "number": input_data["number"],
                "ci_summary": context["ci_summary"],
                "skill_name": context["skill_runtime"]["skill_name"],
                "skill_instructions": context["skill_instructions"],
            }
        )
        return {
            "merge_recommendation": "暂缓合并",
            "risk_points": [],
            "blocking_issues": [],
            "test_suggestions": [],
            "confidence": 0.8,
        }

    monkeypatch.setattr(PRReviewAgent, "run", capture_pr_run)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="targets", full_name="acme/targets")
        db.add(repo)
        db.flush()
        requested_issue = Issue(
            repo_id=repo.id,
            number=142,
            title="Requested issue",
            body="Repro steps and user impact are documented.",
            labels=["bug"],
            state="open",
            created_at=now,
        )
        requested_pr = PullRequest(
            repo_id=repo.id,
            number=86,
            title="Requested PR",
            body="Fixes the requested issue.",
            state="open",
            author="alice",
            created_at=now,
        )
        requested_run = WorkflowRun(
            repo_id=repo.id,
            github_run_id=9001,
            name="requested-ci",
            status="completed",
            conclusion="failure",
            jobs=[{"name": "pytest", "conclusion": "failure"}],
            logs_text="requested test failed",
            created_at=now,
        )
        db.add_all([requested_issue, requested_pr, requested_run])
        db.flush()
        db.add(PRFile(pr_id=requested_pr.id, filename="backend/app/target.py", status="modified", additions=3, deletions=1, patch="target fix"))
        db.commit()

        message = "请用多 Agent 检查 Issue #142、PR #86 和 CI #9001，判断能不能合并。"
        skill_context = {"_skill_registry": get_skill_registry()}
        planning_context, _ = _build_engineering_workflow_runtime(repo, db, message, skill_context)
        spec = WorkflowSpec.model_validate(await PlannerAgent().run({"message": message}, planning_context))
        claims = {task.task_type: task.claim.target_id for task in spec.tasks}
        assert claims["issue_analysis"] == str(requested_issue.id)
        assert claims["pr_review"] == str(requested_pr.id)
        assert claims["ci_debug"] == str(requested_run.id)
        assert "Issue #142" in next(task.objective for task in spec.tasks if task.task_type == "issue_analysis")
        assert "PR #86" in next(task.objective for task in spec.tasks if task.task_type == "pr_review")

        newer_at = now + timedelta(hours=1)
        newer_issue = Issue(repo_id=repo.id, number=999, title="Newer issue", state="open", created_at=newer_at)
        newer_pr = PullRequest(repo_id=repo.id, number=999, title="Newer PR", state="open", created_at=newer_at)
        newer_run = WorkflowRun(
            repo_id=repo.id,
            github_run_id=9999,
            name="newer-ci",
            status="completed",
            conclusion="failure",
            jobs=[],
            created_at=newer_at,
        )
        db.add_all([newer_issue, newer_pr, newer_run])
        db.commit()

        execution_context, runners = _build_engineering_workflow_runtime(repo, db, "确认执行这个计划", skill_context)
        assert execution_context["issue_id"] == str(newer_issue.id)
        assert execution_context["pr_id"] == str(newer_pr.id)
        assert execution_context["workflow_run_id"] == str(newer_run.id)

        report = await WorkflowOrchestrator(runners).run_spec(spec, execution_context)
        summaries = {result["task_type"]: result["summary"] for result in report["task_results"]}
        assert summaries["issue_analysis"].startswith("Issue #142:")
        assert summaries["pr_review"].startswith("PR #86:")
        assert summaries["ci_debug"].startswith("CI requested-ci:")
        assert captured_pr_input["number"] == 86
        assert captured_pr_input["ci_summary"] == {"failed_runs": 2, "latest_failed_run": "requested-ci"}
        assert captured_pr_input["skill_name"] == "pr-review"
        assert "P1/P2" in str(captured_pr_input["skill_instructions"])
        pr_output = next(result["output"] for result in report["task_results"] if result["task_type"] == "pr_review")
        assert pr_output["_skill_runtime"]["validation"] == "entrypoint_completed"


# 用户指定对象不存在时必须报错，不能替换为最新记录。
def test_workflow_rejects_missing_explicit_target_instead_of_using_latest() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="missing", full_name="acme/missing")
        db.add(repo)
        db.flush()
        db.add(PullRequest(repo_id=repo.id, number=1, title="Only PR", state="open"))
        db.commit()

        with pytest.raises(WorkflowTargetResolutionError, match=r"PR #404.*不会回退"):
            _build_engineering_workflow_runtime(repo, db, "请审查 PR #404", {})


# 工作流目标必须属于当前仓库，防止跨项目混用。
@pytest.mark.asyncio
async def test_workflow_rejects_claim_target_from_another_repository() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="one", full_name="acme/one")
        other_repo = Repository(owner="acme", name="two", full_name="acme/two")
        db.add_all([repo, other_repo])
        db.flush()
        local_issue = Issue(repo_id=repo.id, number=1, title="Local issue", state="open")
        foreign_issue = Issue(repo_id=other_repo.id, number=142, title="Foreign issue", state="open")
        db.add_all([local_issue, foreign_issue])
        db.commit()

        workflow_context, runners = _build_engineering_workflow_runtime(repo, db, "检查 Issue", {})
        task = WorkflowTask(
            id="issue_analysis",
            agent_name="issue_analyst_agent",
            task_type="issue_analysis",
            objective="Analyze a tampered claim",
            claim=WorkflowClaim(target_type="issue", target_id=str(foreign_issue.id)),
        )
        spec = WorkflowSpec(goal="tampered", trigger_message="tampered", tasks=[task])

        with pytest.raises(WorkflowTargetResolutionError, match="不属于仓库 acme/one"):
            await runners["issue_analysis"](task, spec, workflow_context)


# 独立任务可以并发，但 Observer 发现阻塞后不能给出合入通过结论。
@pytest.mark.asyncio
async def test_orchestrator_runs_ready_tasks_in_parallel_and_observer_blocks_merge() -> None:
    class StaticPlanner:
        # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check merge readiness",
                trigger_message="merge?",
                tasks=[
                    WorkflowTask(
                        id="pr_review",
                        agent_name="pr_review_agent",
                        task_type="pr_review",
                        objective="Review PR",
                        claim=WorkflowClaim(target_type="pull_request", target_id="pr-1"),
                    ),
                    WorkflowTask(
                        id="ci_debug",
                        agent_name="ci_debug_agent",
                        task_type="ci_debug",
                        objective="Review CI",
                        claim=WorkflowClaim(target_type="workflow_run", target_id="run-1"),
                    ),
                ],
            ).model_dump(mode="json")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_pr(task, spec, context):
        await asyncio.sleep(0.12)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="PR looks mergeable.",
            output={"merge_recommendation": "建议合并", "blocking_issues": []},
            evidence_count=1,
            confidence=0.8,
        )

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_ci(task, spec, context):
        await asyncio.sleep(0.12)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="CI is failing.",
            output={"is_merge_blocking": True, "failure_type": "test", "debug_steps": ["Fix failing tests"]},
            evidence_count=1,
            confidence=0.8,
        )

    started = time.perf_counter()
    report = await WorkflowOrchestrator(
        {"pr_review": run_pr, "ci_debug": run_ci},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "merge?"}, {})
    duration = time.perf_counter() - started

    assert duration < 0.22
    assert report["metrics"]["success_count"] == 2
    assert report["metrics"]["replan_count"] == 0
    finding_types = [item["finding_type"] for item in report["observation"]["findings"]]
    assert "ci_merge_gate" in finding_types
    assert "cross_agent_conflict" in finding_types
    assert "暂缓合并" in report["final_answer"]


# 证据不足时应触发补充 RAG 任务的受限重规划。
@pytest.mark.asyncio
async def test_orchestrator_replans_with_rag_when_evidence_is_missing() -> None:
    class StaticPlanner(PlannerAgent):
        # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check merge readiness",
                trigger_message="merge?",
                tasks=[
                    WorkflowTask(
                        id="pr_review",
                        agent_name="pr_review_agent",
                        task_type="pr_review",
                        objective="Review PR",
                        claim=WorkflowClaim(target_type="pull_request", target_id="pr-1"),
                    )
                ],
            ).model_dump(mode="json")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_pr(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="PR looks mergeable, but no concrete evidence was attached.",
            output={"merge_recommendation": "approve"},
            evidence_count=0,
            confidence=0.72,
        )

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_rag(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="Recovered related historical context.",
            output={"answer": "Found similar PR history."},
            evidence_count=1,
            confidence=0.68,
            citations=[{"type": "pull_request", "id": "pr-0", "title": "Similar PR"}],
        )

    report = await WorkflowOrchestrator(
        {"pr_review": run_pr, "rag_search": run_rag},
        planner=StaticPlanner(),
    ).run({"message": "merge?"}, {"repo_id": "repo-1"})

    assert report["metrics"]["replan_count"] == 1
    assert report["metrics"]["execution_rounds"] == 2
    assert [item["task_id"] for item in report["task_results"]] == ["pr_review", "rag_search"]
    assert "rag_search" in [task["task_type"] for task in report["spec"]["tasks"]]
    assert any(event["type"] == "workflow_replan" for event in report["agent_events"])


# 执行确认计划的入口应遵守其重规划设置，不重新生成另一份初始计划。
@pytest.mark.asyncio
async def test_orchestrator_runs_confirmed_spec_without_replanning() -> None:
    class FailingPlanner:
        # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
        async def run(self, input_data, context):
            raise AssertionError("confirmed specs must not be replanned")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_repo_health(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="Repo baseline collected.",
            evidence_count=1,
            confidence=0.8,
        )

    spec = WorkflowSpec(
        goal="Execute a confirmed plan",
        trigger_message="plan",
        tasks=[
            WorkflowTask(
                id="repo_health",
                agent_name="repo_health_agent",
                task_type="repo_health",
                objective="Collect repo baseline",
                claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
            )
        ],
    )

    report = await WorkflowOrchestrator(
        {"repo_health": run_repo_health},
        planner=FailingPlanner(),  # type: ignore[arg-type]
    ).run_spec(spec.model_dump(mode="json"), {})

    assert report["spec"]["workflow_id"] == spec.workflow_id
    assert report["metrics"]["task_count"] == 1
    assert report["task_results"][0]["task_id"] == "repo_health"


# 卡住的任务需按预算超时，不能无限阻塞整个工作流。
@pytest.mark.asyncio
async def test_orchestrator_times_out_stuck_task() -> None:
    class StaticPlanner:
        # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check stuck worker",
                trigger_message="run",
                tasks=[
                    WorkflowTask(
                        id="slow",
                        agent_name="slow_agent",
                        task_type="slow_task",
                        objective="Never finish quickly",
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    )
                ],
            ).model_dump(mode="json")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_slow(task, spec, context):
        await asyncio.sleep(1)
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary="too late",
        )

    report = await WorkflowOrchestrator(
        {"slow_task": run_slow},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "run"}, {"workflow_task_timeout_seconds": 0.01, "disable_replan": True})

    result = report["task_results"][0]
    assert result["status"] == "error"
    assert result["output"]["error_kind"] == "timeout"


# 前置任务失败时后续依赖项应标记跳过/阻塞，不继续使用无效输入。
@pytest.mark.asyncio
async def test_orchestrator_marks_dependents_skipped_when_dependency_fails() -> None:
    class StaticPlanner:
        # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
        async def run(self, input_data, context):
            return WorkflowSpec(
                goal="Check dependency propagation",
                trigger_message="run",
                tasks=[
                    WorkflowTask(
                        id="root",
                        agent_name="root_agent",
                        task_type="root_task",
                        objective="Fail first",
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    ),
                    WorkflowTask(
                        id="child",
                        agent_name="child_agent",
                        task_type="child_task",
                        objective="Requires root",
                        dependencies=["root"],
                        claim=WorkflowClaim(target_type="repository", target_id="repo-1"),
                    ),
                ],
            ).model_dump(mode="json")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_root(task, spec, context):
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary="root failed",
            error="boom",
        )

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def run_child(task, spec, context):  # pragma: no cover - should not execute
        raise AssertionError("dependent task should be skipped")

    report = await WorkflowOrchestrator(
        {"root_task": run_root, "child_task": run_child},
        planner=StaticPlanner(),  # type: ignore[arg-type]
    ).run({"message": "run"}, {"disable_replan": True})

    statuses = {result["task_id"]: result for result in report["task_results"]}
    assert statuses["root"]["status"] == "error"
    assert statuses["child"]["status"] == "skipped"
    assert statuses["child"]["output"]["error_kind"] == "dependency_failed"


# 要求安全审核却缺失结果时，Observer 应报告问题。
@pytest.mark.asyncio
async def test_observer_flags_missing_safety_review_when_required() -> None:
    spec = WorkflowSpec(
        goal="Draft a GitHub comment",
        trigger_message="comment",
        requires_human_confirmation=True,
        tasks=[],
    )

    observation = await ObserverAgent().run({"spec": spec.model_dump(mode="json"), "task_results": []}, {})

    assert observation["human_review_required"] is True
    assert observation["findings"][0]["finding_type"] == "missing_safety_review"


# 聊天触发工程工作流时需保存父运行及各任务记录，支持回看。
@pytest.mark.asyncio
async def test_chat_workflow_tool_persists_workflow_and_task_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        repo = Repository(owner="acme", name="demo", full_name="acme/demo")
        db.add(repo)
        db.flush()
        conversation = Conversation(repo_id=repo.id, title="Workflow")
        db.add(conversation)
        issue = Issue(
            repo_id=repo.id,
            number=7,
            title="Login API returns 500",
            body="Steps: call login. Expected 401, actual 500. Impact: users cannot login.",
            labels=["bug"],
            state="open",
            created_at=now,
        )
        pr = PullRequest(
            repo_id=repo.id,
            number=8,
            title="Fix login error handling",
            body="Fixes login status handling.",
            state="open",
            author="alice",
            created_at=now,
        )
        db.add_all([issue, pr])
        db.flush()
        db.add(PRFile(pr_id=pr.id, filename="backend/app/auth.py", status="modified", additions=40, deletions=8, patch="handle login"))
        db.add(
            WorkflowRun(
                repo_id=repo.id,
                name="backend-tests",
                status="completed",
                conclusion="failure",
                jobs=[{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
                logs_text="pytest tests/test_auth.py FAILED AssertionError",
                created_at=now,
            )
        )
        db.commit()

        tools = _build_conversation_tools(repo, conversation, db)
        result = await tools["workflow.run_engineering_review"](
            {"message": "这个 PR 能不能合并？请用多 agent 协作检查 Issue、PR 和 CI。"},
            {"team_members": [{"name": "Alice", "role": "Backend", "strengths": "FastAPI auth", "techStack": "Python"}]},
        )

        assert result["route"] == "engineering_workflow"
        assert "多 Agent 协作结论" in result["answer"]
        tool_names = [item["tool_name"] for item in result["tool_calls"]]
        assert "workflow.pr_review" in tool_names
        assert "workflow.ci_debug" in tool_names
        assert db.query(AgentWorkflowRun).count() == 1
        assert db.query(AgentTaskRun).count() >= 3
