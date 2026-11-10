# 学习导读｜HTTP 接口层：仓库会话的列表、创建和删除，复用 conversations 服务处理关联记录。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.conversations import ConversationCreateRequest, ConversationResponse
from app.services.conversations import create_conversation, delete_conversation, list_conversations

router = APIRouter()


# 按 ID 查找仓库，不存在时返回 HTTP 404，避免后续以空仓库继续查询。
def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


# 接口入口：GET /{repo_id}/conversations；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 list[ConversationResponse] 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/{repo_id}/conversations", response_model=list[ConversationResponse])
async def get_conversations(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Any]:
    _repo_or_404(db, repo_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    return conversations


# 接口入口：POST /{repo_id}/conversations；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 ConversationResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/{repo_id}/conversations", response_model=ConversationResponse)
async def add_conversation(
    repo_id: uuid.UUID,
    payload: ConversationCreateRequest | None = None,
    db: Session = Depends(get_db),
) -> Any:
    _repo_or_404(db, repo_id)
    conversation = create_conversation(db, repo_id, payload.title if payload else None)
    db.commit()
    db.refresh(conversation)
    return conversation


# 接口入口：DELETE /{repo_id}/conversations/{conversation_id}；完整 URL 还需加上 main.py 注册的路由前缀。
# 参数：repo_id：当前仓库标识，用于限定数据范围；conversation_id：会话标识，需与当前仓库对应。
# 参数：db：数据库会话，由调用方管理提交与关闭。
@router.delete("/{repo_id}/conversations/{conversation_id}")
async def remove_conversation(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _repo_or_404(db, repo_id)
    replacement = delete_conversation(db, repo_id, conversation_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    return {
        "deleted": str(conversation_id),
        "active_conversation_id": str(replacement.id),
        "conversations": [
            {
                "id": str(item.id),
                "repo_id": str(item.repo_id),
                "title": item.title,
                "status": item.status,
                "message_count": item.message_count,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in conversations
        ],
    }
