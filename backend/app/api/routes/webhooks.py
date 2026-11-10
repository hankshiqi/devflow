# 学习导读｜HTTP 接口层：校验 GitHub Webhook 签名并触发仓库同步，复用同步核心。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

router = APIRouter()


# 用 Webhook Secret 校验请求体签名，确保参与比较的是收到的原始字节。
def verify_signature(body: bytes, signature: str | None) -> None:
    if not settings.github_webhook_secret:
        return
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="缺少 GitHub webhook 签名")
    expected = hmac.new(settings.github_webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    actual = signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=401, detail="GitHub webhook 签名无效")


# 按 owner/repository 全名查找已连接仓库，供 Webhook 事件定位本地记录。
def find_repo_by_full_name(full_name: str) -> Repository | None:
    db: Session = SessionLocal()
    try:
        return db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    finally:
        db.close()


# 识别仓库并复用同步服务；Webhook 本身不是独立的一套实体转换实现。
# 接口入口：POST /github；完整 URL 还需加上 main.py 注册的路由前缀。
@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    verify_signature(body, x_hub_signature_256)
    payload = await request.json()
    repository = payload.get("repository") or {}
    full_name = repository.get("full_name")
    if not full_name:
        return {"accepted": False, "reason": "payload.repository.full_name is missing"}

    repo = find_repo_by_full_name(full_name)
    if repo is None:
        return {"accepted": False, "reason": f"{full_name} is not connected"}

    if x_github_event in {"issues", "pull_request", "workflow_run", "push", "check_suite", "check_run"}:
        #先立即告诉 GitHub：通知已收到然后在后台执行同步
        background_tasks.add_task(sync_repository_by_id, repo.id, RepoSyncRequest(limit=settings.auto_sync_limit))
        return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "sync_scheduled"}

    return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "ignored"}
