# 学习导读｜接口数据契约：会话创建及列表。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# ConversationCreateRequest：定义这一接口对象的字段和类型约束。
class ConversationCreateRequest(BaseModel):
    title: str | None = None


# ConversationResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、repo_id、title。
class ConversationResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    title: str
    status: str = "active"
    message_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
