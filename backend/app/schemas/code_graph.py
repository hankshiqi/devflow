# 学习导读｜接口数据契约：代码符号、关系和影响分析。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# CodeSymbolResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、repo_id、path、name、kind、language、created_at。
class CodeSymbolResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    path: str
    name: str
    kind: str
    language: str
    start_line: int | None = None
    end_line: int | None = None
    branch: str | None = None
    commit_sha: str | None = None
    pr_id: UUID | None = None
    pr_number: int | None = None
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


# CodeGraphSearchResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：query、symbols、relations、documents。
class CodeGraphSearchResponse(BaseModel):
    query: str
    symbols: list[CodeSymbolResponse]
    relations: list[dict]
    documents: list[dict]
