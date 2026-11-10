# 学习导读｜接口数据契约：Skill 描述、资源与 MCP 状态。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from typing import Any

from pydantic import BaseModel, Field


# SkillManifestResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：name、title、version、description、category、entrypoint、output_contract、safety_level。
class SkillManifestResponse(BaseModel):
    name: str
    title: str
    version: str
    description: str
    category: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str


# SkillResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：manifest、path。
class SkillResponse(BaseModel):
    manifest: SkillManifestResponse
    path: str
    instructions: str | None = None


# SkillListResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：skills。
class SkillListResponse(BaseModel):
    skills: list[SkillResponse]


# McpToolResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：name、description。
class McpToolResponse(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


# McpRuntimeResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：status、server_name、server_version、protocol_version、transport。
class McpRuntimeResponse(BaseModel):
    status: str
    server_name: str
    server_version: str
    protocol_version: str
    transport: str
    tools: list[McpToolResponse] = Field(default_factory=list)
