# 学习导读｜接口数据契约：语义搜索请求及命中项。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from uuid import UUID

from pydantic import BaseModel, Field


# SearchRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、query。
class SearchRequest(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    query: str
    source_type: str | None = None
    label: str | None = None
    path: str | None = None
    module: str | None = None
    state: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


# SearchResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：results。
class SearchResponse(BaseModel):
    results: list[dict]
