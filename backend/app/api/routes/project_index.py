# 学习导读｜HTTP 接口层：项目文档扫描状态、手动索引与稍后提醒入口。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.project_index import (
    ProjectIndexScanResponse,
    ProjectIndexSnoozeRequest,
    ProjectIndexSnoozeResponse,
    ProjectIndexStateResponse,
)
from app.services.project_indexing import (
    get_project_index_state,
    mark_project_index_building,
    scan_project_index_by_id,
    snooze_project_index,
)

router = APIRouter()


# 按 ID 查找仓库，不存在时返回 HTTP 404，避免后续以空仓库继续查询。
def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


# 接口入口：GET /{repo_id}/memory-index/state；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 ProjectIndexStateResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/{repo_id}/memory-index/state", response_model=ProjectIndexStateResponse)
async def get_memory_index_state(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    repo = _repo_or_404(db, repo_id)
    return get_project_index_state(db, repo)


# 接口入口：POST /{repo_id}/memory-index/scan；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 ProjectIndexScanResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/memory-index/scan", response_model=ProjectIndexScanResponse, status_code=202)
async def start_memory_index_scan(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> ProjectIndexScanResponse:
    repo = _repo_or_404(db, repo_id)
    mark_project_index_building(db, repo)
    background_tasks.add_task(scan_project_index_by_id, repo.id)
    return ProjectIndexScanResponse(repo_id=repo.id, started=True, status="building")


# 接口入口：POST /{repo_id}/memory-index/snooze；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 ProjectIndexSnoozeResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
#将项目索引提醒延长一段时间不会启动扫描、不会删除索引、也不会停止后端服务。
@router.post("/{repo_id}/memory-index/snooze", response_model=ProjectIndexSnoozeResponse)
async def snooze_memory_index(
    repo_id: uuid.UUID,
    payload: ProjectIndexSnoozeRequest | None = None,
    db: Session = Depends(get_db),
) -> ProjectIndexSnoozeResponse:
    repo = _repo_or_404(db, repo_id)
    row = snooze_project_index(db, repo, days=(payload.days if payload else 7))
    return ProjectIndexSnoozeResponse(repo_id=repo.id, snoozed_until=row.snoozed_until)
