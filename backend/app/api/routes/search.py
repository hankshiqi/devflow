# 学习导读｜HTTP 接口层：仓库范围内的语义搜索 HTTP 入口，实际召回交给 RAG 服务。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()


# 接口入口：POST ；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 SearchResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    filters = {
        "label": payload.label,
        "path": payload.path,
        "module": payload.module,
        "state": payload.state,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }
    results = search_similar_documents(
        db,
        payload.repo_id,
        payload.query,
        source_type=payload.source_type,
        limit=payload.limit,
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    return SearchResponse(results=results)
