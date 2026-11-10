# 学习导读｜接口数据契约：Issue/PR/CI 等 Agent 的结构化分析结果。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from typing import Literal

from pydantic import BaseModel, Field


# DuplicateCandidate：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：issue_id、title。
class DuplicateCandidate(BaseModel):
    issue_id: str
    title: str
    score: float = Field(ge=0, le=1)


# IssueEvidence：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：source_type、title、snippet。
class IssueEvidence(BaseModel):
    source_type: str
    title: str
    snippet: str
    reference: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


# IssueDrafts：定义这一接口对象的字段和类型约束。
class IssueDrafts(BaseModel):
    clarification_comment: str | None = None
    task_breakdown: str | None = None


# IssueAnalysisOutput：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：summary、conclusion、conclusion_reason、category、priority、complexity、suggested_owner、owner_reason。
class IssueAnalysisOutput(BaseModel):
    summary: str
    conclusion: Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]
    conclusion_reason: str
    category: Literal["Bug", "Feature", "Question", "Documentation", "Refactor", "Test", "Ops"]
    priority: Literal["P0", "P1", "P2", "P3"]
    complexity: Literal["S", "M", "L", "XL"]
    suggested_owner: str
    owner_reason: str
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    evidence: list[IssueEvidence] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    drafts: IssueDrafts = Field(default_factory=IssueDrafts)
    confidence: float = Field(ge=0, le=1)


# PRReviewFinding：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：severity、title、evidence、required_action。
class PRReviewFinding(BaseModel):
    severity: Literal["P1", "P2", "P3"]
    title: str
    evidence: str
    required_action: str
    blocking: bool = False


# PRReviewOutput：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：summary。
class PRReviewOutput(BaseModel):
    summary: str
    plan: list[str] = []
    executed_steps: list[str] = []
    merge_recommendation: Literal["建议合入", "修改后合入", "暂缓", "拒绝"] = "暂缓"
    recommendation_reason: str = ""
    review_findings: list[PRReviewFinding] = Field(default_factory=list)
    key_changes: list[str] = []
    risk_points: list[str] = []
    blocking_issues: list[str] = []
    review_checklist: list[str] = []
    test_suggestions: list[str] = []
    files_need_attention: list[str] = []
    review_comments: list[str] = []
    confidence: float = Field(ge=0, le=1)


# CIDebugOutput：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：failure_summary、failure_type。
class CIDebugOutput(BaseModel):
    failure_summary: str
    failure_type: Literal["test", "build", "lint", "dependency", "permission", "environment", "unknown"]
    plan: list[str] = []
    executed_steps: list[str] = []
    first_error: str | None = None
    root_cause: str = ""
    possible_causes: list[str] = []
    fix_steps: list[str] = []
    debug_steps: list[str] = []
    related_files: list[str] = []
    is_merge_blocking: bool = True
    blocking_reason: str = ""
    confidence: float = Field(ge=0, le=1)


# SafetyOutput：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：requires_confirmation、risk_level、reason。
class SafetyOutput(BaseModel):
    requires_confirmation: bool
    risk_level: Literal["low", "medium", "high"]
    reason: str
    draft: str | None = None
