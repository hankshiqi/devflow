# 学习导读｜接口数据契约：多仓库工作区及统计。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field


# WorkspaceCreateRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：name。
class WorkspaceCreateRequest(BaseModel):
    name: str
    description: str | None = None
    repo_ids: list[UUID] = Field(default_factory=list)


# WorkspaceResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、name、created_at。
class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    repo_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# MultiRepoReportRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：start_date、end_date。
class MultiRepoReportRequest(BaseModel):
    workspace_id: UUID | None = None
    repo_ids: list[UUID] = Field(default_factory=list)
    start_date: date
    end_date: date


# MultiRepoReportResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：report_markdown、metrics、repo_summaries。
class MultiRepoReportResponse(BaseModel):
    report_markdown: str
    metrics: dict
    repo_summaries: list[dict]
