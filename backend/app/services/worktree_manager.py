# 学习导读｜为 PR 准备独立 Git worktree 快照，并维护对应代码图及快照清理。
# 快照指向 PR 分支，便于分析当前改动上下文；本地仓库模式不自动创建快照。
"""Git worktree lifecycle for live PR code analysis."""

import json
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository
from app.services.code_analysis import _git_text, _sync_checkout, _sync_code_graph

_SNAPSHOT_LOCKS: dict[str, threading.Lock] = {}
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


# 解析配置中的 checkout 根目录；相对路径基于当前后端工作目录。
def _checkout_root() -> Path:
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


# 按 worktree key 获取互斥锁，避免并发任务同时创建/删除同一快照。
def _snapshot_lock(key: str) -> threading.Lock:
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SNAPSHOT_LOCKS[key] = lock
        return lock


# 对 Git worktree 执行命令并统一捕获命令错误。
def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return result.stdout.strip()


# 确定指定 PR 快照的本地 worktree 目录。
def pr_worktree_path(repo: Repository, pr: PullRequest) -> Path:
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / "worktrees" / str(repo.id) / f"pr-{pr.number}-{safe_name}"


# 返回快照状态元数据文件路径。
def _snapshot_meta_path(path: Path) -> Path:
    return path / ".devflow-pr-snapshot.json"


# 将分支、提交 SHA 和更新时间写入快照元数据。
def _write_snapshot_meta(path: Path, meta: dict[str, Any]) -> None:
    payload = {**meta, "updated_at": datetime.now(timezone.utc).isoformat()}
    _snapshot_meta_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 在 PR 快照上同步代码分析/代码图并返回处理状态。
def _sync_pr_snapshot_analysis(
    db: Session,
    repo: Repository,
    pr: PullRequest,
    snapshot_path: Path,
    branch: str,
    commit_sha: str | None,
) -> int:
    return _sync_code_graph(
        db,
        repo,
        snapshot_path,
        branch=branch,
        commit_sha=commit_sha,
        pr_id=pr.id,
        pr_number=pr.number,
    )


# 参数：message：当前用户问题；pr：已加载的 PR 对象。
def _snapshot_response(
    *,
    status: str,
    message: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    pr: PullRequest,
    snapshot_path: Path | None = None,
    indexed_symbols: int | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "snapshot_name": "PR 分支快照",
        "branch": branch,
        "commit_sha": commit_sha,
        "pr_id": str(pr.id),
        "pr_number": pr.number,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "indexed_symbols": indexed_symbols,
        "fallback": fallback,
        "checkout_kind": "pr_snapshot" if status == "ready" else "default_checkout",
    }


# 为 PR head 分支准备独立 worktree 并扫描代码图；失败时返回回退状态和说明。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo：已加载的仓库对象。
# 参数：pr：已加载的 PR 对象。
def prepare_pr_worktree(db: Session, repo: Repository, pr: PullRequest) -> dict[str, Any]:
    """Prepare a PR branch snapshot for live search and refresh its code graph."""
    lock_key = f"{repo.id}:{pr.id}"
    lock = _snapshot_lock(lock_key)
    with lock:
        # 本地工作区不由服务自动改动分支，因此 PR 快照只支持受管 checkout 模式。
        if (repo.checkout_mode or "managed") == "local":
            return _snapshot_response(status="unsupported", message="本地仓库模式不自动创建或删除 PR 分支快照。", pr=pr, fallback=repo.local_path)
        # 缺少 PR head 分支无法生成精确快照，明确返回降级状态而不猜测默认分支。
        if not pr.head_branch:
            return _snapshot_response(status="missing_head", message="PR 缺少 head branch，继续使用默认分支代码。", pr=pr)
        if not shutil.which("git"):
            return _snapshot_response(status="failed", message="未找到 git，可继续使用默认分支代码。", pr=pr)
        token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
        target = pr_worktree_path(repo, pr)
        try:
            base_checkout = _sync_checkout(repo, token)
            target.parent.mkdir(parents=True, exist_ok=True)
            ref = pr.head_branch
            local_branch = f"devflow/pr-{pr.number}"
            _run_git(["fetch", "origin", ref], cwd=base_checkout)
            # 目标快照已存在时复用/更新它；不存在时才创建隔离 worktree，避免重复分配目录。
            if target.exists():
                _run_git(["fetch", "origin", ref], cwd=target)
                _run_git(["checkout", "-B", local_branch, "FETCH_HEAD"], cwd=target)
            else:
                _run_git(["worktree", "add", "--force", "-B", local_branch, str(target), "FETCH_HEAD"], cwd=base_checkout)
            commit_sha = _git_text(["rev-parse", "HEAD"], target)
            indexed_symbols = _sync_pr_snapshot_analysis(db, repo, pr, target, ref, commit_sha)
            _write_snapshot_meta(
                target,
                {
                    "repo_id": str(repo.id),
                    "pr_id": str(pr.id),
                    "pr_number": pr.number,
                    "branch": ref,
                    "commit_sha": commit_sha,
                    "indexed_symbols": indexed_symbols,
                    "snapshot_kind": "pr_snapshot",
                },
            )
            db.commit()
            return _snapshot_response(
                status="ready",
                branch=ref,
                commit_sha=commit_sha,
                pr=pr,
                snapshot_path=target,
                indexed_symbols=indexed_symbols,
            )
        # Git 操作失败时回滚数据库状态并返回失败快照，调用方仍可选择默认分支分析。
        except Exception as exc:
            db.rollback()
            return _snapshot_response(
                status="fallback",
                message=f"PR 分支快照准备失败，继续使用默认分支代码：{exc}",
                branch=repo.default_branch or "main",
                pr=pr,
            )


# 清理超过保留期的 PR 快照及关联分析记录，避免缓存目录无限增长。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo：已加载的仓库对象。
def cleanup_pr_snapshots(db: Session, repo: Repository, *, max_age_hours: int = 168) -> dict[str, Any]:
    root = _checkout_root() / "worktrees" / str(repo.id)
    if not root.exists():
        return {"removed": 0, "paths": []}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed_paths: list[str] = []
    removed_pr_ids: set[uuid.UUID] = set()
    for path in root.iterdir():
        if not path.is_dir():
            continue
        meta_path = _snapshot_meta_path(path)
        updated_at = None
        snapshot_pr_id: uuid.UUID | None = None
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                updated_raw = metadata.get("updated_at")
                updated_at = datetime.fromisoformat(updated_raw) if updated_raw else None
                if metadata.get("pr_id"):
                    snapshot_pr_id = uuid.UUID(str(metadata["pr_id"]))
            except (OSError, ValueError, json.JSONDecodeError):
                updated_at = None
        if updated_at is None:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if updated_at >= cutoff:
            continue
        try:
            # Remove from Git first so the main checkout does not keep stale worktree records.
            main_checkout = next((item for item in _checkout_root().glob(f"{repo.id}-*") if item.is_dir()), None)
            if main_checkout:
                _run_git(["worktree", "remove", "--force", str(path)], cwd=main_checkout)
            elif path.exists():
                shutil.rmtree(path)
        except Exception:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        removed_paths.append(str(path))
        if snapshot_pr_id:
            removed_pr_ids.add(snapshot_pr_id)
    if removed_pr_ids:
        db.query(CodeRelation).filter(
            CodeRelation.repo_id == repo.id,
            CodeRelation.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
        db.query(CodeSymbol).filter(
            CodeSymbol.repo_id == repo.id,
            CodeSymbol.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
    db.commit()
    return {"removed": len(removed_paths), "paths": removed_paths}
