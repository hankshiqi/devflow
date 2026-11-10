# 学习导读｜接口数据契约：评测请求、执行记录及比较。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from typing import Any

from pydantic import BaseModel, Field
from uuid import UUID


# RagEvalCaseRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：question、reference。
class RagEvalCaseRequest(BaseModel):
    id: str = ""
    question: str
    reference: str
    expected_source_ids: list[str] = Field(default_factory=list)
    source_type: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    kind: str = "positive"
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    answer_regex: str | None = None
    source_file: str | None = None


# EvalRunRequest：定义这一接口对象的字段和类型约束。
class EvalRunRequest(BaseModel):
    name: str = "basic_eval"
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID | None = None
    eval_types: list[str] = Field(default_factory=lambda: ["issue_classification", "rag", "pr_summary", "tool_calls"])
    run_judge: bool = True
    # 期望保留的结果数量；底层通常先超额召回，再去重、筛选和重排。
    top_k: int = Field(default=5, ge=1, le=20)
    suite_id: str | None = None
    cases: list[RagEvalCaseRequest] | None = None


# EvalIssueDraftRequest：定义这一接口对象的字段和类型约束。
class EvalIssueDraftRequest(BaseModel):
    title: str | None = None


# EvalRunResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：eval_id、status、result。
class EvalRunResponse(BaseModel):
    eval_id: str
    status: str
    result: dict
