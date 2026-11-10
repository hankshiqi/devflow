# 学习导读｜接口数据契约：操作草稿、确认结果与审计记录。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ActionDraftCreateRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、content。
class ActionDraftCreateRequest(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    draft_type: str = Field(default="issue_comment")
    target_type: str = Field(default="issue")
    target_id: UUID | None = None
    title: str = ""
    content: str
    risk_level: str = "medium"
    metadata: dict = Field(default_factory=dict)


# ActionDraftResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、repo_id、draft_type、target_type、title、content、risk_level、status、created_at。
class ActionDraftResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    draft_type: str
    target_type: str
    target_id: UUID | None = None
    title: str
    content: str
    risk_level: str
    status: str
    execution_result: dict | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


# ActionDraftConfirmResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：draft、audit_id。
class ActionDraftConfirmResponse(BaseModel):
    draft: ActionDraftResponse
    audit_id: UUID


# AuditLogResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、action、status、request_json、result_json、created_at。
class AuditLogResponse(BaseModel):
    id: UUID
    user_id: UUID | None = None
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID | None = None
    action: str
    target_type: str | None = None
    target_id: str | None = None
    status: str
    request_json: dict
    result_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}
