# 学习导读｜接口数据契约：聊天请求、回答、工具记录与推荐提问。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from uuid import UUID

from pydantic import BaseModel, Field


# 聊天输入，指定仓库、会话、问题和客户端上下文；服务端仍要校验数据范围。
# 未设默认值的字段需由调用方提供：repo_id、message。
class ChatRequest(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    conversation_id: UUID | None = None
    message: str
    context: dict = Field(default_factory=dict)


# ChatPlanExecuteRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：workflow_spec。
class ChatPlanExecuteRequest(ChatRequest):
    workflow_spec: dict


# ToolCallTrace：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：tool_name、arguments、status。
class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: dict
    status: str
    error: str | None = None
    error_kind: str | None = None
    retryable: bool | None = None
    attempts: int | None = None
    tool_use_id: str | None = None
    skill_name: str | None = None
    skill_title: str | None = None
    skill_version: str | None = None
    skill_steps: list[str] = Field(default_factory=list)
    skill_activation_mode: str | None = None
    skill_activation_reason: str | None = None
    skill_instruction_digest: str | None = None


# 聊天最终结果，把回答、工具轨迹和引用一起返回页面。
# 未设默认值的字段需由调用方提供：run_id、message_id、route、tool_calls、answer。
class ChatResponse(BaseModel):
    run_id: UUID | str
    message_id: UUID | str
    route: str
    tool_calls: list[ToolCallTrace]
    agent_events: list[dict] = []
    answer: str
    # 证据来源引用，帮助追踪结论依据；引用存在不自动证明结论正确。
    citations: list[dict] = []
    completion: dict = {}
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}


# ChatHistoryMessage：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、role、content。
class ChatHistoryMessage(BaseModel):
    id: UUID | str
    conversation_id: UUID | str | None = None
    role: str
    content: str
    route: str | None = None
    tool_calls: list[ToolCallTrace] = []
    agent_events: list[dict] = []
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}
    created_at: str | None = None


# PromptRecommendationRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id。
class PromptRecommendationRequest(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    exclude: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=4, le=5)


# PromptRecommendationResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：agent、suggestions。
class PromptRecommendationResponse(BaseModel):
    agent: str
    suggestions: list[str]
