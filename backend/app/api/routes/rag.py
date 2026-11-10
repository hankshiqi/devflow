# 学习导读｜HTTP 接口层：知识库配置、文档上传/处理、片段、召回测试、问答和重建索引的 HTTP 入口。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
import asyncio
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    Document,
    KnowledgeSourceDocument,
    Repository,
    RetrievalTestRun,
)
from app.db.session import get_db
from app.schemas.rag import (
    KnowledgeBaseConfigUpdate,
    KnowledgeBaseResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentResponse,
    ModelCatalogResponse,
    RAGAnswerResponse,
    RAGQueryRequest,
    RAGReindexResponse,
    RAGSearchResponse,
    RetrievalTestHistoryResponse,
    RetrievalTestRequest,
    RetrievalTestResponse,
)
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES, process_knowledge_document
from app.services.rag.embeddings import validate_embedding_contract
from app.services.rag.knowledge_base import (
    config_dict,
    get_or_create_config,
    model_catalog,
    record_retrieval_test,
    retrieve_knowledge,
    runtime_status,
)
from app.services.rag.qa import answer_question, build_sources, rag_status, reindex_documents
from app.services.rag.vector_store import delete_document_group

router = APIRouter()


# 按 ID 查找仓库，不存在时返回 HTTP 404，避免后续以空仓库继续查询。
def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None or repo.provider == "knowledge_base":
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


# 清理上传文件名中的路径等不安全部分，避免将客户端文件名直接当磁盘路径。
def _safe_filename(filename: str) -> str:
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


# 定位并准备上传文件存储目录，供文档记录保存实际路径。
def _upload_root(repo_id: uuid.UUID) -> Path:
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


# 校验切片大小与重叠参数，阻止无法推进或过度重叠的配置。
def _validate_chunking(chunk_size: int, overlap: int) -> None:
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="切片重叠必须小于切片长度")


# 把知识源 ORM 行转换为上传文档 API 响应。
def _document_response(item: KnowledgeSourceDocument) -> KnowledgeDocumentResponse:
    return KnowledgeDocumentResponse(
        id=item.id,
        repo_id=item.repo_id,
        name=item.name,
        source_type=item.source_type,
        status=item.status,
        content_type=item.content_type,
        file_size=item.file_size,
        character_count=item.character_count,
        chunk_count=item.chunk_count,
        error_message=item.error_message,
        created_at=item.created_at,
        updated_at=item.updated_at,
        completed_at=item.completed_at,
    )


# 参数：db：数据库会话，由调用方管理提交与关闭；repo：已加载的仓库对象。
def _knowledge_base(db: Session, repo: Repository) -> KnowledgeBaseResponse:
    config = get_or_create_config(db, repo.id)
    document_count, character_count = (
        db.query(func.count(KnowledgeSourceDocument.id), func.coalesce(func.sum(KnowledgeSourceDocument.character_count), 0))
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .one()
    )
    chunk_count = db.query(func.count(Document.id)).filter(Document.repo_id == repo.id).scalar() or 0
    return KnowledgeBaseResponse(
        id=repo.id,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description or "",
        provider=repo.provider or "github",
        document_count=int(document_count or 0),
        chunk_count=int(chunk_count),
        character_count=int(character_count or 0),
        config=config_dict(config),
        runtime=runtime_status(config),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )


# 接口入口：GET /models；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 ModelCatalogResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/models", response_model=ModelCatalogResponse)
async def get_model_catalog() -> ModelCatalogResponse:
    return ModelCatalogResponse.model_validate(model_catalog())


# 接口入口：GET /repositories；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[KnowledgeBaseResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/repositories", response_model=list[KnowledgeBaseResponse])
async def list_rag_repositories(db: Session = Depends(get_db)) -> list[KnowledgeBaseResponse]:
    repos = (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )
    return [_knowledge_base(db, repo) for repo in repos]


# 接口入口：GET /repositories/{repo_id}；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 KnowledgeBaseResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/repositories/{repo_id}", response_model=KnowledgeBaseResponse)
async def get_rag_repository(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeBaseResponse:
    return _knowledge_base(db, _repo_or_404(db, repo_id))


# 接口入口：PATCH /repositories/{repo_id}/config；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 KnowledgeBaseResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；db：数据库会话，由调用方管理提交与关闭。
@router.patch("/repositories/{repo_id}/config", response_model=KnowledgeBaseResponse)
async def update_rag_config(
    repo_id: uuid.UUID,
    payload: KnowledgeBaseConfigUpdate,
    db: Session = Depends(get_db),
) -> KnowledgeBaseResponse:
    repo = _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id, commit=False)
    changes = payload.model_dump(exclude_unset=True)
    next_size = int(changes.get("chunk_size", config.chunk_size))
    next_overlap = int(changes.get("chunk_overlap", config.chunk_overlap))
    _validate_chunking(next_size, next_overlap)
    try:
        validate_embedding_contract(
            str(changes.get("embedding_provider", config.embedding_provider)),
            str(changes.get("embedding_model", config.embedding_model)),
            int(changes.get("embedding_dimensions", config.embedding_dimensions)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    for key, value in changes.items():
        setattr(config, key, value)
    # 只有磁盘文件写入成功后才提交数据库；异常路径回滚记录并尝试删除文件副本。
    db.commit()
    db.refresh(repo)
    return _knowledge_base(db, repo)


# 接口入口：GET /repositories/{repo_id}/documents；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[KnowledgeDocumentResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/repositories/{repo_id}/documents", response_model=list[KnowledgeDocumentResponse])
async def list_documents(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeDocumentResponse]:
    _repo_or_404(db, repo_id)
    items = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo_id)
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .all()
    )
    return [_document_response(item) for item in items]


# 接口入口：POST /repositories/{repo_id}/documents；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 KnowledgeDocumentResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；db：数据库会话，由调用方管理提交与关闭。
@router.post("/repositories/{repo_id}/documents", response_model=KnowledgeDocumentResponse, status_code=202)
async def upload_document(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    _repo_or_404(db, repo_id)
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="文件不能超过 12 MB")
    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    # 上传前先按扩展名白名单拒绝当前解析器不支持的文件类型。
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"暂不支持 {suffix or '无扩展名'} 文件")
    checksum = hashlib.sha256(data).hexdigest()
    duplicate = (
        db.query(KnowledgeSourceDocument)
        .filter(
            KnowledgeSourceDocument.repo_id == repo_id,
            KnowledgeSourceDocument.checksum == checksum,
            KnowledgeSourceDocument.status != "failed",
        )
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .first()
    )
    # 相同内容指纹已经存在时返回原知识源，避免重复保存和重复索引。
    if duplicate is not None:
        return _document_response(duplicate)
    document = KnowledgeSourceDocument(
        repo_id=repo_id,
        name=filename,
        content_type=file.content_type,
        checksum=checksum,
        file_size=len(data),
        stored_path="pending",
        status="queued",
    )
    db.add(document)
    # 先写入知识源记录并取得 ID，再以该 ID 生成磁盘路径和后续关联的文档片段。
    db.flush()
    stored_path = _upload_root(repo_id) / f"{document.id}-{filename}"
    try:
        await asyncio.to_thread(stored_path.write_bytes, data)
        document.stored_path = str(stored_path)
        # 只有磁盘文件写入成功后才提交数据库；异常路径回滚记录并尝试删除文件副本。
        db.commit()
    except Exception:
        db.rollback()
        try:
            await asyncio.to_thread(stored_path.unlink, missing_ok=True)
        except OSError:
            pass
        raise
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


# 接口入口：GET /repositories/{repo_id}/documents/{document_id}/chunks；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[KnowledgeChunkResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；document_id：上传源文件或目标记录的标识，含义以所查模型为准。
# 参数：db：数据库会话，由调用方管理提交与关闭。
@router.get("/repositories/{repo_id}/documents/{document_id}/chunks", response_model=list[KnowledgeChunkResponse])
async def list_document_chunks(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[KnowledgeChunkResponse]:
    _repo_or_404(db, repo_id)
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_id == document_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        KnowledgeChunkResponse(
            id=row.id,
            position=int((row.meta or {}).get("chunk_index", index)) + 1,
            title=row.title,
            content=row.content,
            character_count=len(row.content),
        )
        for index, row in enumerate(rows)
    ]


# 接口入口：POST /repositories/{repo_id}/documents/{document_id}/retry；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 KnowledgeDocumentResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/repositories/{repo_id}/documents/{document_id}/retry", response_model=KnowledgeDocumentResponse)
async def retry_document(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    _repo_or_404(db, repo_id)
    document = db.get(KnowledgeSourceDocument, document_id)
    if document is None or document.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    document.status = "queued"
    document.error_message = None
    # 只有磁盘文件写入成功后才提交数据库；异常路径回滚记录并尝试删除文件副本。
    db.commit()
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


# 接口入口：DELETE /repositories/{repo_id}/documents/{document_id}；完整 URL 还需加上 main.py 注册的路由前缀。
@router.delete("/repositories/{repo_id}/documents/{document_id}")
async def delete_document(repo_id: uuid.UUID, document_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    _repo_or_404(db, repo_id)
    source = db.get(KnowledgeSourceDocument, document_id)
    if source is None or source.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    delete_document_group(db, repo_id, document_id, source_type="knowledge_file", commit=False)
    path = Path(source.stored_path)
    if path.exists() and path.is_file():
        path.unlink()
    db.delete(source)
    # 只有磁盘文件写入成功后才提交数据库；异常路径回滚记录并尝试删除文件副本。
    db.commit()
    return {"id": str(document_id), "deleted": True}


# 接口入口：POST /repositories/{repo_id}/retrieval-tests；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RetrievalTestResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；db：数据库会话，由调用方管理提交与关闭。
@router.post("/repositories/{repo_id}/retrieval-tests", response_model=RetrievalTestResponse)
async def run_retrieval_test(
    repo_id: uuid.UUID,
    payload: RetrievalTestRequest,
    db: Session = Depends(get_db),
) -> RetrievalTestResponse:
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.query.strip(),
        top_k=payload.top_k,
        retrieval_method=payload.retrieval_method,
        rerank_enabled=payload.rerank_enabled,
        score_threshold_enabled=payload.score_threshold_enabled,
        score_threshold=payload.score_threshold,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    run = record_retrieval_test(db, repo_id, payload.query.strip(), trace, results, duration_ms)
    return RetrievalTestResponse(
        id=run.id,
        query=run.query,
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
        created_at=run.created_at,
    )


# 接口入口：GET /repositories/{repo_id}/retrieval-tests；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[RetrievalTestHistoryResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；limit：本次返回数量上限。
# 参数：db：数据库会话，由调用方管理提交与关闭。
@router.get("/repositories/{repo_id}/retrieval-tests", response_model=list[RetrievalTestHistoryResponse])
async def list_retrieval_tests(
    repo_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RetrievalTestHistoryResponse]:
    _repo_or_404(db, repo_id)
    runs = (
        db.query(RetrievalTestRun)
        .filter(RetrievalTestRun.repo_id == repo_id)
        .order_by(RetrievalTestRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrievalTestHistoryResponse(
            id=run.id,
            query=run.query,
            result_count=len(run.results_json or []),
            took_ms=run.duration_ms,
            created_at=run.created_at,
        )
        for run in runs
    ]


# 接口入口：GET /{repo_id}/status；完整 URL 还需加上 main.py 注册的路由前缀。
@router.get("/{repo_id}/status")
async def get_rag_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    return {**rag_status(db, repo_id), "config": config_dict(config), "runtime": runtime_status(config)}


# 接口入口：POST /{repo_id}/search；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RAGSearchResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/search", response_model=RAGSearchResponse)
async def search_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGSearchResponse:
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.question.strip(),
        top_k=payload.top_k,
        source_types=payload.source_types,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    return RAGSearchResponse(
        query=payload.question.strip(),
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
    )


# 接口入口：POST /{repo_id}/ask；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RAGAnswerResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/ask", response_model=RAGAnswerResponse)
async def ask_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGAnswerResponse:
    _repo_or_404(db, repo_id)
    result = await answer_question(
        db,
        repo_id, #当前知识库所属仓库
        payload.question, #问题 
        payload.top_k, #设置的top_k
        payload.source_types, #限制来源类型
        payload.metadata_filters, #元数据筛选条件
    )
    return RAGAnswerResponse.model_validate(result)


# 接口入口：POST /{repo_id}/reindex；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RAGReindexResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/reindex", response_model=RAGReindexResponse)
async def reindex_rag(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> RAGReindexResponse:
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    count = reindex_documents(db, repo_id)
    return RAGReindexResponse(
        repo_id=repo_id,
        indexed_documents=count,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
    )
