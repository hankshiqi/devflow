# 学习导读｜接口数据契约：PR、变更文件与审查评论。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import PRReviewOutput


# PRFileResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、filename、status、additions、deletions。
class PRFileResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    additions: int
    deletions: int
    # PR 的增删行差异文本，可能为空，不能等同完整源码。
    patch: str | None = None

    model_config = {"from_attributes": True}


# PRReviewCommentResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id。
class PRReviewCommentResponse(BaseModel):
    id: UUID
    body: str | None = None
    path: str | None = None
    line: int | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# PullRequestResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、repo_id、number、title、state。
class PullRequestResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    number: int
    title: str
    body: str | None = None
    state: str
    author: str | None = None
    base_branch: str | None = None
    head_branch: str | None = None
    merged_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    files: list[PRFileResponse] = []
    review_comments: list[PRReviewCommentResponse] = []

    model_config = {"from_attributes": True}


# PRAnalysisResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：analysis_id、result。
class PRAnalysisResponse(BaseModel):
    analysis_id: UUID | str
    result: PRReviewOutput
