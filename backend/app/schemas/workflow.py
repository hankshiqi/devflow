# 学习导读｜接口数据契约：工作流计划、任务依赖、执行状态和观察结果。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]


# 生成新的 UUID 工作流标识，供没有外部指定 ID 的计划使用。
def _workflow_id() -> str:
    return f"workflow_{uuid.uuid4().hex[:12]}"


# 声明任务要分析的目标、允许的证据类型和是否有写入意图，防止执行时切换对象或越界。
# 未设默认值的字段需由调用方提供：target_type。
class WorkflowClaim(BaseModel):
    target_type: str
    target_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    write_intent: bool = False
    reason: str = ""


# 一个可执行分析任务：包含执行器类型、目标、前置任务和是否属于关键任务。
# 未设默认值的字段需由调用方提供：id、agent_name、task_type、objective、claim。
class WorkflowTask(BaseModel):
    id: str
    agent_name: str
    task_type: str
    objective: str
    # 前置任务 ID 列表；依赖未完成前不能将该任务当作可运行。
    dependencies: list[str] = Field(default_factory=list)
    claim: WorkflowClaim
    input_hint: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False


# 完整工作流计划；tasks 描述任务依赖，max_parallel_tasks 限定同时执行数量。
# 未设默认值的字段需由调用方提供：goal、trigger_message。
class WorkflowSpec(BaseModel):
    workflow_id: str = Field(default_factory=_workflow_id)
    goal: str
    trigger_message: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    tasks: list[WorkflowTask] = Field(default_factory=list)
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)
    requires_human_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# 单任务执行结果，状态、证据数量、引用和错误与业务输出分别保存。
# 未设默认值的字段需由调用方提供：task_id、agent_name、task_type、status。
class WorkflowTaskResult(BaseModel):
    task_id: str
    agent_name: str
    task_type: str
    status: WorkflowTaskStatus
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    confidence: float | None = Field(default=None, ge=0, le=1)
    # 证据来源引用，帮助追踪结论依据；引用存在不自动证明结论正确。
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# Observer 检查发现的问题，严重程度决定是否需要阻塞或人工复核。
# 未设默认值的字段需由调用方提供：finding_type、severity、message。
class WorkflowFinding(BaseModel):
    finding_type: str
    severity: WorkflowFindingSeverity
    message: str
    task_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""


# 对一轮任务结果的整体检查，聚合发现、置信度和人工审核要求。
class WorkflowObservation(BaseModel):
    observer_agent: str = "observer_agent"
    findings: list[WorkflowFinding] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    human_review_required: bool = False
    summary: str = ""


# 最终工作流报告，把原始计划、任务结果、检查发现与回答放在一起。
# 未设默认值的字段需由调用方提供：spec、task_results、observation、final_answer。
class WorkflowReport(BaseModel):
    spec: WorkflowSpec
    task_results: list[WorkflowTaskResult]
    observation: WorkflowObservation
    final_answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    # 证据来源引用，帮助追踪结论依据；引用存在不自动证明结论正确。
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)
