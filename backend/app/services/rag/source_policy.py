# 学习导读｜定义哪些来源进入语义检索，以及哪些内容必须由结构化查询或工作区工具读取。
# 调用写入/检索入口时都应用来源策略，避免只在界面上限制而底层仍混入源码。
from collections.abc import Iterable
from typing import Literal


RetrievalMode = Literal["rag", "direct"]

# RAG is reserved for reusable, unstructured project knowledge where semantic
# recall is useful. Structured live facts remain in their source tables/APIs.
RAG_SOURCE_TYPES = frozenset(
    {
        "issue",
        "pull_request",
        "workflow_run",
        "project_doc",
        "knowledge_file",
        "memory_note",
    }
)

DIRECT_SOURCE_TYPES = frozenset(
    {
        "project_overview",
        "project_manifest",
        "team_member",
        "weekly_report",
        "chat_session",
        "conversation_memory",
        "thread_memory",
    }
)


# 按来源类型和配置决定是否允许语义/关键词检索。
def retrieval_mode(source_type: str) -> RetrievalMode:
    if source_type in RAG_SOURCE_TYPES:
        return "rag"
    return "direct"


# 整理一个或多个来源类型，移除空值并统一后续筛选输入。
def normalize_source_types(source_types: Iterable[str | None] | None) -> list[str]:
    normalized = [str(source_type).strip() for source_type in (source_types or []) if source_type and str(source_type).strip()]
    return list(dict.fromkeys(normalized))


# 按来源类型决定是否生成并存储向量，落实源码/结构化资料与 RAG 的边界。
def should_vectorize(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


# 判断来源是否允许进入当前 RAG 搜索范围。
def is_searchable_source(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


# 为文档附加来源检索策略标记，便于检查索引边界。
def policy_metadata(source_type: str) -> dict[str, object]:
    mode = retrieval_mode(source_type)
    return {
        "retrieval_mode": mode,
        "rag_enabled": mode == "rag",
    }
