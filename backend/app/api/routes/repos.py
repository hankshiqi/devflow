# 学习导读｜HTTP 接口层：仓库连接、列表、选择、同步与删除，支持已有本地目录和受管 checkout。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, AnalysisResult, AuditLog, ChatFeedback, ChatMessage, ChatSession, CodeRelation, CodeSymbol, Conversation, ConversationMemory, Document, EvidenceItem, Issue, KnowledgeBaseConfig, KnowledgeGraphEdge, KnowledgeSourceDocument, MemoryCandidate, PRFile, PRReviewComment, ProjectIndexState, PullRequest, RecallEvent, Repository, RetrievalTestRun, ThreadMemory, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.repos import RepoConnectRequest, RepoConnectResponse, RepoResponse, RepoSyncRequest, RepoSyncResponse
from app.core.config import settings
from app.core.security import encrypt_token
from app.services.github.client import GitHubClient
from app.services.code_analysis import sync_repository_code_analysis_by_id
from app.services.rag.vector_store import delete_milvus_documents
from app.services.repos_sync import sync_repository

router = APIRouter()

SUPPORTED_CONNECT_PROVIDERS = {"github", "github_compatible"}
GITHUB_HOSTS = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class GitRemoteInfo:
    owner: str
    repo: str
    clone_url: str
    host: str | None = None
    default_branch: str | None = None


# 清理远程地址或用户输入中的空白和控制字符。
def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


# 解析用户提供的本地路径并验证目录有效性。
def _resolve_user_path(raw_path: str, field_name: str) -> Path:
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


# 在指定目录运行只读 Git 命令并返回标准输出。
def _git_output(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="未找到 git 可执行文件，无法检查本地仓库。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=400, detail=f"Failed to inspect local repository: {detail}") from exc
    return result.stdout.strip()


# 解析常见 HTTPS/SSH Git 远程地址中的主机、所有者和仓库名。
def _parse_git_remote_url(remote_url: str) -> GitRemoteInfo:
    value = remote_url.strip()
    host: str | None = None
    path = ""
    scp_like = re.match(r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<path>.+)$", value)

    if scp_like and "://" not in value:
        host = scp_like.group("host")
        path = scp_like.group("path")
    else:
        parsed = urlparse(value)
        host = parsed.hostname or parsed.netloc or None
        path = parsed.path

    parts = [item.strip() for item in path.replace("\\", "/").strip("/").split("/") if item.strip()]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")

    owner = parts[-2]
    repo = re.sub(r"\.git$", "", parts[-1], flags=re.IGNORECASE)
    if not owner or not repo:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")
    return GitRemoteInfo(owner=owner, repo=repo, clone_url=value, host=host.lower() if host else None)


# 检查本地路径是不是 Git 仓库并读取当前分支、远程地址等信息。
def _inspect_local_repository(raw_path: str) -> tuple[Path, GitRemoteInfo]:
    path = _resolve_user_path(raw_path, "local_path")
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Local repository path does not exist or is not a directory: {path}")

    inside_work_tree = _git_output(["rev-parse", "--is-inside-work-tree"], path).lower()
    if inside_work_tree != "true":
        raise HTTPException(status_code=400, detail=f"Local path is not inside a git repository: {path}")

    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], path)).resolve()
    remote_url = _git_output(["remote", "get-url", "origin"], repo_root)
    remote_info = _parse_git_remote_url(remote_url)
    current_branch = _git_output(["branch", "--show-current"], repo_root) or None
    return repo_root, GitRemoteInfo(
        owner=remote_info.owner,
        repo=remote_info.repo,
        clone_url=remote_info.clone_url,
        host=remote_info.host,
        default_branch=current_branch,
    )


# 根据显式选择或远程主机推断 GitHub/Gitee 等代码托管服务。
def _infer_provider(requested_provider: str | None, remote_host: str | None) -> str:
    provider = requested_provider or "github"
    if remote_host and remote_host not in GITHUB_HOSTS and provider == "github":
        return "github_compatible"
    return provider


# 按提供方和自定义地址确定最终 API 根地址。
def _effective_api_base_url(provider: str, raw_base_url: str | None, remote_host: str | None) -> str | None:
    if provider == "github":
        return "https://api.github.com"
    base_url = _clean_text(raw_base_url)
    if base_url:
        return base_url.rstrip("/")
    if remote_host and remote_host not in GITHUB_HOSTS:
        return f"https://{remote_host}/api/v3"
    return None


# 把父目录与仓库名组合成克隆目标路径，并处理路径冲突。
def _clone_target_from_parent(raw_parent: str, repo_name: str) -> Path:
    parent = _resolve_user_path(raw_parent, "clone_parent_dir")
    if parent.exists() and not parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Download parent path is not a directory: {parent}")

    target = (parent / repo_name).resolve()
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Download target exists and is not a directory: {target}")
    if target.exists() and not (target / ".git").exists() and any(target.iterdir()):
        raise HTTPException(
            status_code=400,
            detail=f"Download target already exists and is not an empty git repository: {target}",
        )
    return target


# 接口入口：POST /connect；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RepoConnectResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/connect", response_model=RepoConnectResponse)
async def connect_repo(payload: RepoConnectRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> RepoConnectResponse:
    local_path = _clean_text(payload.local_path)
    clone_parent_dir = _clean_text(payload.clone_parent_dir)
    owner = _clean_text(payload.owner)
    repo_name = _clean_text(payload.repo)
    requested_provider = payload.provider or "github"
    token = payload.token or payload.github_token
    checkout_path: Path | None = None
    checkout_mode = "managed"
    remote_info: GitRemoteInfo | None = None

    # 连接入口先判断本地路径模式；本地仓库的 owner/name/remote 信息从 Git 配置解析。
    if local_path:
        #检查本地路径是不是 Git 仓库
        checkout_path, remote_info = _inspect_local_repository(local_path)
        owner = remote_info.owner
        repo_name = remote_info.repo
        checkout_mode = "local"
    elif not owner or not repo_name:
        raise HTTPException(status_code=400, detail="Provide either a local repository path or owner/repository.")

    # 依据用户选择和本地 Git remote 主机推断服务提供方，再据此确定兼容 API 地址。
    provider = _infer_provider(requested_provider, remote_info.host if remote_info else None)
    base_url = _effective_api_base_url(provider, payload.api_base_url, remote_info.host if remote_info else None)
    if provider not in SUPPORTED_CONNECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"{provider} real connection is not implemented yet. Use demo mode or github_compatible with a GitHub-compatible REST API.",
        )

    if not local_path and clone_parent_dir:
        checkout_path = _clone_target_from_parent(clone_parent_dir, repo_name or "repository")

    full_name = f"{owner}/{repo_name}"
    effective_demo_mode = payload.demo_mode
    fallback_message: str | None = None
    repo_data = {
        "id": None,
        "description": "Demo repository for DevFlow AI",
        "default_branch": remote_info.default_branch if remote_info and remote_info.default_branch else "main",
        "clone_url": remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git",
    }
    # 非演示模式才访问远程仓库 API；API 失败时只在允许的 demo fallback 条件下继续。
    if not payload.demo_mode:
        try:
            repo_data = await GitHubClient(token, base_url).get_repo(owner or "", repo_name or "")
        except Exception as exc:
            if token or not settings.demo_mode:
                raise HTTPException(status_code=400, detail=f"Repository connection failed: {exc}") from exc
            effective_demo_mode = True
            fallback_message = (
                f"GitHub verification failed without a token: {exc}. "
                "Added the repository in local cache mode; add a token or retry later for full GitHub sync."
            )

    repo = db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    if repo is None:
        repo = Repository(owner=owner or "", name=repo_name or "", full_name=full_name)
        db.add(repo)
    repo.owner = owner or repo.owner
    repo.name = repo_name or repo.name
    repo.provider = provider
    repo.api_base_url = base_url
    if token:
        repo.github_token_encrypted = encrypt_token(token, settings.token_encryption_key)
    repo.description = repo_data.get("description")
    repo.default_branch = repo_data.get("default_branch")
    repo.clone_url = repo_data.get("clone_url") or (remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git")
    repo.local_path = str(checkout_path) if checkout_path else None
    repo.checkout_mode = checkout_mode
    repo.github_id = repo_data.get("id")
    # 把仓库连接信息写入本地数据库后再刷新 ORM 对象，后续后台同步使用这个持久化配置。
    db.commit()
    db.refresh(repo)
    sync_result: RepoSyncResponse | None = None
    if not effective_demo_mode:
        sync_result = await sync_repository(
            db,
            repo,
            RepoSyncRequest(
                sync_issues=payload.sync_issues,
                sync_pull_requests=payload.sync_pull_requests,
                sync_workflow_runs=payload.sync_workflow_runs,
                limit=payload.initial_sync_limit,
            ),
        )
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    elif fallback_message:
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return RepoConnectResponse(
        repo_id=repo.id,
        full_name=repo.full_name,
        provider=repo.provider,
        api_base_url=repo.api_base_url,
        clone_url=repo.clone_url,
        local_path=repo.local_path,
        checkout_mode=repo.checkout_mode,
        default_branch=repo.default_branch,
        connected=True,
        demo_mode=effective_demo_mode,
        message=(
            fallback_message
            or (
                "Demo mode only created a local repository record."
                if effective_demo_mode
                else "Repository connection verified. Initial Issue/PR/CI sync completed; code checkout and graph analysis started."
            )
        ),
        sync_status=sync_result.status if sync_result else ("fallback" if fallback_message else "skipped"),
        synced=sync_result.synced if sync_result else {"issues": 0, "pull_requests": 0, "workflow_runs": 0},
    )


# 接口入口：GET ；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[RepoResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("", response_model=list[RepoResponse])
async def list_repos(db: Session = Depends(get_db)) -> list[Repository]:
    return (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )


# 接口入口：POST /{repo_id}/sync；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 RepoSyncResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/sync", response_model=RepoSyncResponse)
async def sync_repo(
    repo_id: UUID,
    payload: RepoSyncRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RepoSyncResponse:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    result = await sync_repository(db, repo, payload)
    background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return result


# 接口入口：POST /{repo_id}/select；完整 URL 还需加上 main.py 注册的路由前缀。
@router.post("/{repo_id}/select")
async def select_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    repo.updated_at = datetime.now(timezone.utc)
    # 把仓库连接信息写入本地数据库后再刷新 ORM 对象，后续后台同步使用这个持久化配置。
    db.commit()
    return {"repo_id": str(repo.id), "full_name": repo.full_name, "selected": True}


# 接口入口：DELETE /{repo_id}；完整 URL 还需加上 main.py 注册的路由前缀。
# 参数：repo_id：当前仓库标识，用于限定数据范围；db：数据库会话，由调用方管理提交与关闭。
@router.delete("/{repo_id}")
async def delete_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # 删除前先收集仓库下游记录 ID，供没有直接 repo_id 的分析、会话和工作流记录定向清理。
    issue_ids = [item.id for item in db.query(Issue.id).filter(Issue.repo_id == repo.id).all()]
    pr_ids = [item.id for item in db.query(PullRequest.id).filter(PullRequest.repo_id == repo.id).all()]
    run_ids = [item.id for item in db.query(WorkflowRun.id).filter(WorkflowRun.repo_id == repo.id).all()]
    conversation_ids = [item.id for item in db.query(Conversation.id).filter(Conversation.repo_id == repo.id).all()]
    session_ids = [
        item.id
        for item in db.query(ChatSession.id)
        .filter(
            or_(
                ChatSession.repo_id == repo.id,
                ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id)
        .filter(
            or_(
                AgentWorkflowRun.repo_id == repo.id,
                AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]

    for target_type, ids in [
        ("issue", issue_ids),
        ("pull_request", pr_ids),
        ("workflow_run", run_ids),
    ]:
        if ids:
            db.query(AnalysisResult).filter(
                AnalysisResult.target_type == target_type,
                AnalysisResult.target_id.in_(ids),
            ).delete(synchronize_session=False)

    knowledge_sources = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .all()
    )
    # 先记下上传文件路径，数据库记录提交删除后再清理磁盘文件。
    stored_paths = [Path(source.stored_path) for source in knowledge_sources]

    if pr_ids:
        db.query(PRFile).filter(PRFile.pr_id.in_(pr_ids)).delete(synchronize_session=False)
        db.query(PRReviewComment).filter(PRReviewComment.pr_id.in_(pr_ids)).delete(synchronize_session=False)
    # 先删除关系库中的仓库文档和关联索引记录；Milvus 是外部存储，下面另行尝试清理。
    db.query(Document).filter(Document.repo_id == repo.id).delete(synchronize_session=False)
    db.query(RetrievalTestRun).filter(RetrievalTestRun.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeSourceDocument).filter(KnowledgeSourceDocument.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ActionDraft).filter(ActionDraft.repo_id == repo.id).delete(synchronize_session=False)
    db.query(AuditLog).filter(AuditLog.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(
        or_(
            MemoryCandidate.repo_id == repo.id,
            MemoryCandidate.conversation_id.in_(conversation_ids) if conversation_ids else False,
            MemoryCandidate.session_id.in_(session_ids) if session_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(RecallEvent).filter(
        or_(
            RecallEvent.repo_id == repo.id,
            RecallEvent.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(
        or_(
            AgentWorkflowRun.repo_id == repo.id,
            AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatMessage).filter(
        or_(
            ChatMessage.repo_id == repo.id,
            ChatMessage.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatSession).filter(
        or_(
            ChatSession.repo_id == repo.id,
            ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(
        or_(
            ConversationMemory.repo_id == repo.id,
            ConversationMemory.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(
        or_(
            EvidenceItem.repo_id == repo.id,
            EvidenceItem.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(AgentRun).filter(
        or_(
            AgentRun.repo_id == repo.id,
            AgentRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(Issue).filter(Issue.repo_id == repo.id).delete(synchronize_session=False)
    db.query(PullRequest).filter(PullRequest.repo_id == repo.id).delete(synchronize_session=False)
    db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).delete(synchronize_session=False)
    for workspace in db.query(Workspace).all():
        next_repo_ids = [item for item in (workspace.repo_ids or []) if str(item) != str(repo.id)]
        if next_repo_ids != (workspace.repo_ids or []):
            workspace.repo_ids = next_repo_ids
    # 清理完依赖仓库的表记录后标记仓库本身删除，随后统一提交本地数据库事务。
    db.delete(repo)
    # 把仓库连接信息写入本地数据库后再刷新 ORM 对象，后续后台同步使用这个持久化配置。
    db.commit()

    # 数据库提交后以非严格模式删除 Milvus 向量；外部存储失败会作为清理结果报告，不能回滚已提交 SQL。
    milvus_deleted = delete_milvus_documents(repo_id=repo.id, strict=False)
    removed_files = 0
    file_cleanup_errors: list[str] = []
    # 逐个移除上传资料的磁盘副本；单个文件删除失败会记录下来，不中断其他清理项。
    for path in stored_paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed_files += 1
        except OSError as exc:
            file_cleanup_errors.append(f"{path}: {exc}")

    return {
        "repo_id": str(repo_id),
        "deleted": True,
        "cleanup": {
            "milvus_deleted": milvus_deleted,
            "removed_files": removed_files,
            "file_errors": file_cleanup_errors,
        },
    }


# 接口入口：GET /{repo_id}/sync-status；完整 URL 还需加上 main.py 注册的路由前缀。
@router.get("/{repo_id}/sync-status")
async def sync_status(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return {
        "repo_id": str(repo.id),
        "status": "completed" if repo.last_sync_at else "not_started",
        "last_sync_at": repo.last_sync_at,
        "message": "最近一次同步已成功完成。" if repo.last_sync_at else "仓库还没有同步过。",
    }
