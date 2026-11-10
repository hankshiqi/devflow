# 学习导读｜HTTP 接口层：代码符号图的重建、列表和搜索入口，连接本地代码分析服务。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.code_graph import CodeGraphSearchResponse, CodeSymbolResponse
from app.services.code_analysis import sync_repository_code_analysis
from app.services.code_search import read_code_excerpt, repository_checkout_path
from app.services.permissions import require_permission
from app.services.worktree_manager import pr_worktree_path

router = APIRouter()


# 接口入口：POST /repos/{repo_id}/code-graph/rebuild；完整 URL 还需加上 main.py 注册的路由前缀。
@router.post("/repos/{repo_id}/code-graph/rebuild")
async def rebuild_code_graph(
    repo_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    symbol_count = await sync_repository_code_analysis(db, repo)
    return {"repo_id": str(repo.id), "indexed_symbols": symbol_count}


# 接口入口：GET /repos/{repo_id}/code-symbols；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[CodeSymbolResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/repos/{repo_id}/code-symbols", response_model=list[CodeSymbolResponse])
async def list_code_symbols(
    repo_id: UUID,
    path: str | None = None,
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[CodeSymbol]:
    # 所有代码符号查询先限定仓库，后续 PR 条件只是进一步收窄范围。
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if path:
        query = query.filter(CodeSymbol.path.contains(path))
    # PR 查询可补充对应快照/checkout 的文件内容；符号图本身来自预先扫描的本地源码。
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    return query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()


# 接口入口：GET /repos/{repo_id}/code-graph/search；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 CodeGraphSearchResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
# 参数：repo_id：当前仓库标识，用于限定数据范围；limit：本次返回数量上限。
# 参数：db：数据库会话，由调用方管理提交与关闭。
@router.get("/repos/{repo_id}/code-graph/search", response_model=CodeGraphSearchResponse)
async def search_code_graph(
    repo_id: UUID,
    q: str = Query(default=""),
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    text = q.strip()
    # 所有代码符号查询先限定仓库，后续 PR 条件只是进一步收窄范围。
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    # PR 查询可补充对应快照/checkout 的文件内容；符号图本身来自预先扫描的本地源码。
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    if text:
        query = query.filter(or_(CodeSymbol.name.contains(text), CodeSymbol.path.contains(text)))
    symbols = query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()
    symbol_ids = [item.id for item in symbols]
    relations = []
    # 先根据匹配符号 ID 取关系边，再将关系转换成前端展示所需的节点和边结构。
    if symbol_ids:
        rows = (
            db.query(CodeRelation)
            .filter(
                CodeRelation.repo_id == repo_id,
                CodeRelation.pr_id == pr_id if pr_id else True,
                CodeRelation.pr_number == pr_number if pr_number else True,
                or_(CodeRelation.source_symbol_id.in_(symbol_ids), CodeRelation.target_symbol_id.in_(symbol_ids), CodeRelation.target_name.in_([item.name for item in symbols])),
            )
            .limit(limit * 4)
            .all()
        )
        relations = [
            {
                "id": str(row.id),
                "source_symbol_id": str(row.source_symbol_id) if row.source_symbol_id else None,
                "target_symbol_id": str(row.target_symbol_id) if row.target_symbol_id else None,
                "source_name": row.source_name,
                "target_name": row.target_name,
                "relation_type": row.relation_type,
                "path": row.path,
                "branch": row.branch,
                "pr_id": str(row.pr_id) if row.pr_id else None,
                "pr_number": row.pr_number,
                "metadata": row.meta or {},
            }
            for row in rows
        ]
    documents = []
    checkout_path = repository_checkout_path(repo)
    selected_pr = None
    # PR 查询可补充对应快照/checkout 的文件内容；符号图本身来自预先扫描的本地源码。
    if pr_id:
        selected_pr = db.get(PullRequest, pr_id)
    elif pr_number:
        selected_pr = (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == repo_id, PullRequest.number == pr_number)
            .one_or_none()
        )
    if selected_pr and selected_pr.repo_id == repo_id:
        snapshot_path = pr_worktree_path(repo, selected_pr)
        if snapshot_path.exists():
            checkout_path = snapshot_path
    if checkout_path.exists():
        # 仅对有限数量的命中符号读取源码片段，控制响应大小并保持路径/行号可定位。
        for symbol in symbols[:limit]:
            snippet = read_code_excerpt(checkout_path, symbol.path, symbol.start_line, line_count=7)
            if not snippet:
                continue
            documents.append(
                {
                    "id": f"workspace:{symbol.path}:{symbol.start_line}",
                    "title": f"{symbol.path}:{symbol.start_line}",
                    "source_id": symbol.path,
                    "path": symbol.path,
                    "symbol": symbol.name,
                    "start_line": symbol.start_line,
                    "end_line": symbol.start_line + 6,
                    "pr_id": str(selected_pr.id) if selected_pr else None,
                    "pr_number": selected_pr.number if selected_pr else None,
                    "snippet": snippet[:1200],
                }
            )
    return {"query": text, "symbols": symbols, "relations": relations, "documents": documents}
