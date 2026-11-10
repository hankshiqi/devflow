# 学习导读｜上传文档的后台处理：解析 → 切片 → 批量 Embedding → 写入索引 → 更新状态。
# 使用独立数据库 Session，避免后台任务复用已结束请求的会话；失败时记录错误供前端查看。
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import KnowledgeSourceDocument
from app.db.session import SessionLocal
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.knowledge_base import get_or_create_config
from app.services.rag.vector_store import DocumentPayload, add_documents, delete_document_group

SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown", ".mdx", ".pdf", ".docx", ".json", ".csv", ".log"}
EMBEDDING_BATCH_SIZE = 64


# 按扩展名分派 TXT/Markdown/PDF/DOCX 等文档解析器。
def extract_text(filename: str, data: bytes) -> str:
    return extract_document_text(filename, data)


# 按固定批大小调用 Embedding，减少一次请求过大并保持片段与向量顺序对应。
def _embed_in_batches(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_knowledge_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
        )
    return vectors


# 用独立 Session 更新上传文档处理状态，供前端轮询查看进度与错误。
def _set_status(document_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
    # 后台任务/状态更新独立管理数据库会话，结束时在 finally 关闭。
    db = SessionLocal()
    try:
        document = db.get(KnowledgeSourceDocument, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error
        document.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


# 后台执行文件解析、结构切片、批量编码及索引替换，失败回滚 SQL 并保存错误状态。
def process_knowledge_document(document_id: uuid.UUID) -> None:
    _set_status(document_id, "parsing")
    # 后台任务/状态更新独立管理数据库会话，结束时在 finally 关闭。
    db = SessionLocal()
    try:
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is None:
            return
        with Path(source.stored_path).open("rb") as handle:
            data = handle.read(MAX_DOCUMENT_BYTES + 1)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"knowledge file exceeds the {MAX_DOCUMENT_BYTES} byte limit")
        text = extract_text(source.name, data)
        if not text.strip():
            raise ValueError("文档中没有可索引的文本")

        # 阶段状态先提交，便于前端轮询看到正在切片，而不是一直停留在上传中。
        source.status = "splitting"
        source.character_count = len(text)
        db.commit()
        config = get_or_create_config(db, source.repo_id)
        chunks = chunk_document(
            text,
            source.name,
            max_chars=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        if not chunks:
            raise ValueError("文档切分后没有有效片段")

        # 切片完成后批量生成向量；向量的顺序必须与 titles/contents 一一对应。
        source.status = "embedding"
        source.chunk_count = len(chunks)
        db.commit()
        titles = []
        for index, chunk in enumerate(chunks):
            label = chunk.get("section_title")
            if not label and chunk.get("page"):
                label = f"Page {chunk['page']}"
            label = str(label or f"Chunk {index + 1}")
            child_suffix = f" ({int(chunk.get('child_index') or 0) + 1})" if chunk.get("child_index") else ""
            titles.append(f"{source.name} · {label}{child_suffix}")
        contents = [str(chunk["content"]) for chunk in chunks]
        vectors = _embed_in_batches(
            [f"{title}\n{content}" for title, content in zip(titles, contents)],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        payloads = [
            DocumentPayload(
                source_type="knowledge_file",
                source_id=source.id,
                title=title,
                content=content,
                metadata={
                    **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                    "display_title": source.name,
                    "filename": source.name,
                    "knowledge_document_id": str(source.id),
                    "checksum": source.checksum,
                    "file_size": source.file_size,
                    "content_type": source.content_type,
                    "chunk_index": index,
                    "chunk_chars": len(content),
                    "total_chunks": len(chunks),
                    "chunk_strategy": "structure_aware_v1",
                },
            )
            for index, (title, content, chunk) in enumerate(zip(titles, contents, chunks))
        ]
        # 重试或重建同一源文件时先清理旧片段；这是替换，不是简单追加。
        delete_document_group(db, source.repo_id, source.id, source_type="knowledge_file", commit=False)
        add_documents(
            db,
            source.repo_id,
            payloads,
            commit=False,
            vectors=vectors,
            embedding_info={
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
            },
        )
        source.status = "completed"
        source.error_message = None
        source.completed_at = datetime.now(timezone.utc)
        source.updated_at = source.completed_at
        db.commit()
    except Exception as exc:
        # 撤销本次尚未提交的 SQL 修改，再另行提交 failed 状态；外部向量库不随 SQL 自动回滚。
        db.rollback()
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is not None:
            source.status = "failed"
            source.error_message = str(exc)[:1000]
            source.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
