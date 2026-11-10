# 学习导读｜接口数据契约：项目索引状态及扫描操作。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ProjectIndexStateResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、status。
class ProjectIndexStateResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    status: str
    fingerprint: str = ""
    docs_indexed: int = 0
    docs_total: int = 0
    error_message: str | None = None
    summary_json: dict[str, Any] | None = None
    snoozed_until: datetime | None = None
    last_scan_at: datetime | None = None


# ProjectIndexScanResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、started。
class ProjectIndexScanResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    started: bool
    status: str = "building"


# ProjectIndexSnoozeRequest：定义这一接口对象的字段和类型约束。
class ProjectIndexSnoozeRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


# ProjectIndexSnoozeResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、snoozed_until。
class ProjectIndexSnoozeResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    snoozed_until: datetime
