# 学习导读｜接口数据契约：知识条目、团队、长期记忆与知识图谱。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# MemoryNoteRequest：定义这一接口对象的字段和类型约束。
class MemoryNoteRequest(BaseModel):
    title: str = ""
    content: str = ""


# TeamMemberRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：name。
class TeamMemberRequest(BaseModel):
    name: str
    role: str = ""
    strengths: str = ""
    tech_stack: str = Field(default="", alias="techStack")

    model_config = {"populate_by_name": True}


# TeamMemberResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、name。
class TeamMemberResponse(BaseModel):
    id: UUID
    name: str
    role: str = ""
    strengths: str = ""
    tech_stack: str = Field(default="", alias="techStack")
    created_at: datetime | None = None

    model_config = {"populate_by_name": True}


# KnowledgeItemResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、document_ids、source_type、title、content_preview、chunk_count。
class KnowledgeItemResponse(BaseModel):
    id: UUID
    document_ids: list[UUID]
    source_type: str
    title: str
    content_preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int
    created_at: datetime | None = None


# KnowledgeUploadResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、filename、chunks、indexed。
class KnowledgeUploadResponse(BaseModel):
    id: UUID
    filename: str
    chunks: int
    indexed: bool
    duplicate: bool = False


# KnowledgeGraphNodeResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、type、title。
class KnowledgeGraphNodeResponse(BaseModel):
    id: str
    type: str
    title: str
    subtitle: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlighted: bool = False


# KnowledgeGraphEdgeResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、from_type、from_id、to_type、to_id、relation。
class KnowledgeGraphEdgeResponse(BaseModel):
    id: str
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    relation: str
    confidence: float = 1.0
    source: str = "explicit"
    metadata: dict[str, Any] = Field(default_factory=dict)


# KnowledgeGraphResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、nodes、edges。
class KnowledgeGraphResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    center_type: str | None = None
    center_id: str | None = None
    depth: int = 1
    nodes: list[KnowledgeGraphNodeResponse]
    edges: list[KnowledgeGraphEdgeResponse]
    stats: dict[str, Any] = Field(default_factory=dict)


# KnowledgeGraphRebuildResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、edges、nodes。
class KnowledgeGraphRebuildResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    edges: int
    nodes: int


# RecallEventResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、tool_name。
class RecallEventResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    tool_name: str
    query: str = ""
    scope: str = "project"
    mode: str = "hybrid"
    status: str = "success"
    result_count: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    # 证据来源引用，帮助追踪结论依据；引用存在不自动证明结论正确。
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


# MemorySearchResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：query、results、result_count。
class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    result_count: int


# MemoryStatusResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id。
class MemoryStatusResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    index: dict[str, Any] = Field(default_factory=dict)
    documents: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    recall: dict[str, Any] = Field(default_factory=dict)
    candidates: dict[str, Any] = Field(default_factory=dict)
    vector: dict[str, Any] = Field(default_factory=dict)


# MemoryCandidateResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、kind、title、content、status、source。
class MemoryCandidateResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    session_id: UUID | None = None
    kind: str
    title: str
    content: str
    status: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    reviewed_at: datetime | None = None


# MemoryCandidateActionResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、status。
class MemoryCandidateActionResponse(BaseModel):
    id: UUID
    status: str
    # 文档片段标识，用于定位一条检索记录，与 source_id 的粒度不同。
    document_id: UUID | None = None
