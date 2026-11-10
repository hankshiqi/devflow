# 学习导读｜接口数据契约：仓库连接、同步和状态。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# 仓库连接参数，区分远程身份/地址、本地目录和首次同步策略。
class RepoConnectRequest(BaseModel):
    owner: str | None = None
    repo: str | None = None
    token: str | None = None
    github_token: str | None = None
    provider: str = "github"
    api_base_url: str | None = None
    local_path: str | None = None
    clone_parent_dir: str | None = None
    demo_mode: bool = False
    initial_sync_limit: int = Field(default=5, ge=1, le=50)
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_workflow_runs: bool = True


# RepoResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、owner、name、full_name。
class RepoResponse(BaseModel):
    id: UUID
    owner: str
    name: str
    full_name: str
    provider: str = "github"
    api_base_url: str | None = None
    clone_url: str | None = None
    local_path: str | None = None
    checkout_mode: str = "managed"
    description: str | None = None
    default_branch: str | None = None
    last_sync_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# RepoConnectResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、full_name、provider、default_branch、connected、demo_mode、message。
class RepoConnectResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    full_name: str
    provider: str
    api_base_url: str | None = None
    clone_url: str | None = None
    local_path: str | None = None
    checkout_mode: str = "managed"
    default_branch: str | None
    connected: bool
    demo_mode: bool
    message: str
    sync_status: str = "not_started"
    synced: dict[str, int] | None = None


# 同步开关和数量限制，控制本次读取 Issue、PR 与 CI 的范围。
class RepoSyncRequest(BaseModel):
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_workflow_runs: bool = True
    limit: int = 30


# 本次同步结果，返回各实体处理数量和同步时间。
# 未设默认值的字段需由调用方提供：repo_id、status、synced。
class RepoSyncResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    status: str
    synced: dict[str, int]
