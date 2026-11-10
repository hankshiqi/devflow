# 学习导读｜把同步后的 Issue、PR 讨论、失败 CI 日志等允许来源转换为 RAG 文档片段。
# 索引是结构化实体到检索文本的投影；原始状态仍以业务表为准。
import re

from sqlalchemy.orm import Session, selectinload

from app.db.models import Issue, KnowledgeBaseConfig, PullRequest, Repository, WorkflowRun
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.vector_store import DocumentPayload, replace_documents

SYNCED_REPO_SOURCE_TYPES = ["issue", "pull_request", "workflow_run"]
CI_LOG_CHUNK_CHARS = 8000
CI_LOG_MAX_CHUNKS = 6


# 清理 CI 日志中的不适合索引内容，降低噪声并保留失败线索。
def sanitize_ci_log(text: str) -> str:
    sanitized = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sanitized = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]", sanitized)
    sanitized = re.sub(r"\bAKIA[A-Z0-9]{16}\b", "[REDACTED_AWS_ACCESS_KEY]", sanitized)
    sanitized = re.sub(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
        "[REDACTED_JWT]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)\b(authorization\s*:\s*bearer)\s+[^\s,;]+",
        r"\1 [REDACTED]",
        sanitized,
    )
    sanitized = re.sub(r"(?i)(::add-mask::)[^\r\n]+", r"\1[REDACTED]", sanitized)
    return re.sub(
        r"(?i)\b([a-z0-9_]*(?:api[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token|"
        r"token|secret|password|passwd|private[_-]?key|client[_-]?secret|account[_-]?key)[a-z0-9_]*)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        sanitized,
    )


# 把较长 CI 日志按错误附近片段切开，作为可检索的历史证据。
def _ci_log_chunks(text: str) -> tuple[list[str], bool]:
    sanitized = sanitize_ci_log(text.strip())
    max_chars = CI_LOG_CHUNK_CHARS * CI_LOG_MAX_CHUNKS
    bounded = sanitized[:max_chars]
    chunks = [
        bounded[start : start + CI_LOG_CHUNK_CHARS]
        for start in range(0, len(bounded), CI_LOG_CHUNK_CHARS)
        if bounded[start : start + CI_LOG_CHUNK_CHARS].strip()
    ]
    return chunks, len(sanitized) > max_chars


# 将时间转换为 ISO 字符串，供 JSON 输出或 metadata 保存。
def _iso(value) -> str | None:
    return value.isoformat() if value else None


# 把同步业务实体转换为统一 RAG 文档载荷。
def _payload(source_type: str, source_id, title: str, content: str, metadata: dict) -> DocumentPayload | None:
    if not content.strip():
        return None
    return DocumentPayload(
        source_type=source_type,
        source_id=source_id,
        title=title[:500],
        content=content,
        metadata=metadata,
    )


# 把仓库允许进入 RAG 的业务文本转成 DocumentPayload，再按来源替换旧索引。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
def index_repository_documents(
    db: Session,
    repo_id,
    *,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> int:
    repo = db.get(Repository, repo_id)
    if repo is None:
        return 0

    payloads: list[DocumentPayload] = []

    # 从本地已同步的结构化数据开始构建候选，不在索引函数里直接请求 GitHub。
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    for issue in issues:
        content = "\n".join([issue.title, issue.body or ""])
        item = _payload(
            "issue",
            issue.id,
            issue.title,
            content,
            {
                "number": issue.number,
                "labels": issue.labels or [],
                "state": issue.state,
                "author": issue.author,
                "assignees": issue.assignees or [],
                "created_at": _iso(issue.created_at),
                "updated_at": _iso(issue.updated_at),
                "closed_at": _iso(issue.closed_at),
            },
        )
        if item:
            payloads.append(item)

    # PR 查询同时预加载文件和行级评论，随后把讨论与变更摘要组合成可检索文本。
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    for pr in prs:
        comments_text = "\n".join(
            f"{comment.path or ''}:{comment.line or ''} {comment.author or ''}: {comment.body or ''}"
            for comment in pr.review_comments
        )
        content = "\n".join(
            [
                pr.title,
                pr.body or "",
                "review comments:",
                comments_text,
            ]
        )
        item = _payload(
            "pull_request",
            pr.id,
            pr.title,
            content,
            {
                "number": pr.number,
                "state": pr.state,
                "author": pr.author,
                "files": [file.filename for file in pr.files],
                "base_branch": pr.base_branch,
                "head_branch": pr.head_branch,
                "created_at": _iso(pr.created_at),
                "updated_at": _iso(pr.updated_at),
                "merged_at": _iso(pr.merged_at),
            },
        )
        if item:
            payloads.append(item)

    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    for run in runs:
        # 只把存在失败结论和日志正文的 CI 运行纳入知识索引，避免成功流水线噪声。
        if run.conclusion != "failure" or not (run.logs_text or "").strip():
            continue
        log_chunks, logs_truncated = _ci_log_chunks(run.logs_text or "")
        for chunk_index, log_chunk in enumerate(log_chunks):
            content = "\n".join([run.name, log_chunk])
            item = _payload(
                "workflow_run",
                run.id,
                run.name if len(log_chunks) == 1 else f"{run.name}#{chunk_index + 1}",
                content,
                {
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "html_url": run.html_url,
                    "jobs": [job.get("name") for job in (run.jobs or []) if isinstance(job, dict)],
                    "created_at": _iso(run.created_at),
                    "updated_at": _iso(run.updated_at),
                    "chunk_index": chunk_index,
                    "total_chunks": len(log_chunks),
                    "logs_truncated": logs_truncated,
                    "secrets_redacted": True,
                },
            )
            if item:
                payloads.append(item)

    vectors = None
    embedding_info = None
    # Embedding 配置只有在请求生成向量时读取；纯文本替换流程可以跳过模型调用。
    config = (
        db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_id).one_or_none()
        if generate_embeddings
        else None
    )
    if config is not None and payloads:
        # 将所有候选正文按同一模型和维度批量生成向量，列表顺序需与 payloads 保持一致。
        vectors = embed_knowledge_texts(
            [f"{payload.title}\n{payload.content}" for payload in payloads],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
        embedding_info = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
        }

    # 按来源类型整体替换旧索引并写入新内容；数据库提交由调用方控制，便于与同步事务配合。
    docs = replace_documents(
        db,
        repo_id,
        SYNCED_REPO_SOURCE_TYPES,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    return len(docs)
