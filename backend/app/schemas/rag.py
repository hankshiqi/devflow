# 学习导读｜接口数据契约：知识库配置、上传文档、检索、问答和模型目录。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# 知识库局部更新请求；只提交需要修改的配置，具体更新语义见对应路由。
class KnowledgeBaseConfigUpdate(BaseModel):
    embedding_provider: str | None = None
    embedding_model: str | None = None
    # 文档/问题向量的维度，必须匹配当前集合，不能靠截断或补零适配。
    embedding_dimensions: int | None = Field(default=None, ge=64, le=4096)
    # vector 为语义向量，fulltext 为关键词 BM25，hybrid 合并两路候选。
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None
    rerank_provider: str | None = None
    rerank_model: str | None = None
    # 期望保留的结果数量；底层通常先超额召回，再去重、筛选和重排。
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold_enabled: bool | None = None
    # 检索/重排分数门槛；不是回答正确概率，并非所有打分方式天然可比。
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    # 片段目标字符预算，标题和结构信息也会影响实际分块。
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    # 相邻片段保留的重叠量；还需满足小于切片预算的跨字段约束。
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


# 知识库概要与运行状态；文档数量、片段数量和配置分别返回。
# 未设默认值的字段需由调用方提供：id、name、full_name。
class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    full_name: str
    description: str = ""
    provider: str = "github"
    document_count: int = 0
    chunk_count: int = 0
    character_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


# 上传源文件的状态和统计；源文件 ID 与拆出的片段 ID 不同。
# 未设默认值的字段需由调用方提供：id、repo_id、name、source_type、status。
class KnowledgeDocumentResponse(BaseModel):
    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    name: str
    source_type: str
    status: str
    content_type: str | None = None
    file_size: int = 0
    character_count: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


# 源文件中的一个检索片段，包含正文、位置和可用状态。
# 未设默认值的字段需由调用方提供：id、position、title、content、character_count。
class KnowledgeChunkResponse(BaseModel):
    id: UUID
    position: int
    title: str
    content: str
    character_count: int
    enabled: bool = True


# 问答/检索请求：问题、Top-K、来源类型与元数据筛选条件。
class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # 期望保留的结果数量；底层通常先超额召回，再去重、筛选和重排。
    top_k: int = Field(default=5, ge=1, le=10)
    source_types: list[str] = Field(default_factory=list, max_length=20)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


# 一条候选证据，区分引用编号、片段 ID、源对象 ID 和排序分数。
# 未设默认值的字段需由调用方提供：citation、document_id、source_id、source_type、title、snippet、score。
class RAGSourceResponse(BaseModel):
    citation: int
    # 文档片段标识，用于定位一条检索记录，与 source_id 的粒度不同。
    document_id: str
    # 原始来源对象的标识；一个来源可拆出多个 Document。
    source_id: str
    source_type: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)


# 只检索的结果，返回候选与配置/耗时，不等于已完成答案生成。
# 未设默认值的字段需由调用方提供：query、results、result_count、took_ms。
class RAGSearchResponse(BaseModel):
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)


# 召回测试参数，可临时覆盖知识库的检索方式、重排与阈值。
class RetrievalTestRequest(BaseModel):
    query: str = Field(min_length=1, max_length=250)
    # 期望保留的结果数量；底层通常先超额召回，再去重、筛选和重排。
    top_k: int | None = Field(default=None, ge=1, le=20)
    # vector 为语义向量，fulltext 为关键词 BM25，hybrid 合并两路候选。
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None
    score_threshold_enabled: bool | None = None
    # 检索/重排分数门槛；不是回答正确概率，并非所有打分方式天然可比。
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


# 一次召回测试的命中项、配置和耗时，便于检查排序与过滤。
# 未设默认值的字段需由调用方提供：id、query、results、result_count、took_ms。
class RetrievalTestResponse(BaseModel):
    id: UUID
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


# RetrievalTestHistoryResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、query、result_count、took_ms。
class RetrievalTestHistoryResponse(BaseModel):
    id: UUID
    query: str
    result_count: int
    took_ms: int
    created_at: datetime | None = None


# ModelCatalogResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：embedding_models、rerank_models、qwen。
class ModelCatalogResponse(BaseModel):
    embedding_models: list[dict[str, Any]]
    rerank_models: list[dict[str, Any]]
    qwen: dict[str, Any]


# 完整问答结果，包含答案、编号来源、生成方式和证据门控说明。
# 未设默认值的字段需由调用方提供：question、answer、sources、retrieval_count、generation_mode、took_ms。
class RAGAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSourceResponse]
    retrieval_count: int
    # 说明由模型生成、证据抽取还是没有足够证据，避免把兜底当模型回答。
    generation_mode: Literal["llm", "extractive", "no_evidence"]
    model: str | None = None
    answer_gate: dict[str, Any] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    took_ms: int


# 知识库状态，分别展示文本数量、Embedding 契约、向量存储和生成能力。
# 未设默认值的字段需由调用方提供：repo_id、document_count、source_count。
class RAGStatusResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    document_count: int
    source_count: int
    source_types: dict[str, int] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    supported_files: list[str] = Field(default_factory=list)


# 重建索引后的文档数量和实际 Embedding 契约，便于核对新旧索引。
# 未设默认值的字段需由调用方提供：repo_id、indexed_documents、embedding_provider、embedding_model。
class RAGReindexResponse(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    indexed_documents: int
    embedding_provider: str
    embedding_model: str
