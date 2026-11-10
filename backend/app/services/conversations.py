# 学习导读｜维护仓库下的会话创建、默认会话、标题、活跃时间和关联记录清理。
# 仓库与会话同时限定消息范围，不能只靠会话标题定位数据。
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    ChatFeedback,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    EvidenceItem,
    MemoryCandidate,
    RecallEvent,
    Repository,
)


# 从仓库名或首条消息生成默认会话标题。
def _default_title(db: Session, repo_id: uuid.UUID) -> str:
    existing = db.query(Conversation).filter(Conversation.repo_id == repo_id).count()
    return "默认会话" if existing == 0 else f"新会话 {existing + 1}"


# 为仓库查找或创建默认会话，并确保它可供聊天入口使用。
def ensure_default_conversation(db: Session, repo_id: uuid.UUID) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.created_at.asc())
        .first()
    )
    if conversation:
        return conversation
    conversation = Conversation(repo_id=repo_id, title="默认会话", status="active")
    db.add(conversation)
    db.flush()
    return conversation


# 校验指定会话属于当前仓库；未指定时使用该仓库默认会话。
def ensure_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID | None = None) -> Conversation:
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
            .one_or_none()
        )
        if conversation:
            return conversation
    return ensure_default_conversation(db, repo_id)


# 按仓库范围返回会话列表并附带最近活动信息。
def list_conversations(db: Session, repo_id: uuid.UUID) -> list[Conversation]:
    ensure_default_conversation(db, repo_id)
    _backfill_repo_conversation(db, repo_id)
    db.flush()
    return (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .all()
    )


# 创建仓库下的新会话及其默认标题/初始状态。
def create_conversation(db: Session, repo_id: uuid.UUID, title: str | None = None) -> Conversation:
    conversation = Conversation(repo_id=repo_id, title=(title or "").strip() or _default_title(db, repo_id))
    db.add(conversation)
    db.flush()
    return conversation


# 删除会话及其关联的消息/记忆等记录，按实际关联关系清理避免留下孤立数据。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：conversation_id：会话标识，需与当前仓库对应。
def delete_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
        .one_or_none()
    )
    if conversation is None:
        return ensure_default_conversation(db, repo_id)

    session_ids = [
        item.id
        for item in db.query(ChatSession.id).filter(ChatSession.conversation_id == conversation.id).all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id).filter(AgentWorkflowRun.conversation_id == conversation.id).all()
    ]

    db.query(ChatFeedback).filter(ChatFeedback.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.conversation_id == conversation.id).delete(synchronize_session=False)
    if session_ids:
        db.query(MemoryCandidate).filter(MemoryCandidate.session_id.in_(session_ids)).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.conversation_id == conversation.id).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(AgentWorkflowRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(AgentRun).filter(AgentRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(EvidenceItem.conversation_id == conversation.id).delete(synchronize_session=False)
    conversation.status = "deleted"
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()

    replacement = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .first()
    )
    return replacement or create_conversation(db, repo_id, "默认会话")


# 更新会话最近活动时间，并按调用方要求增加消息计数。
def touch_conversation(
    db: Session,
    conversation: Conversation,
    *,
    user_message: str | None = None,
    increment: int = 0,
) -> None:
    if user_message and conversation.message_count == 0 and conversation.title.startswith(("默认会话", "新会话")):
        title = user_message.strip().replace("\n", " ")
        if title:
            conversation.title = title[:40]
    conversation.message_count = (conversation.message_count or 0) + increment
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()


# 为缺少默认会话的已有仓库批量补建记录。
def backfill_all_default_conversations(db: Session) -> None:
    for repo in db.query(Repository).all():
        ensure_default_conversation(db, repo.id)
        _backfill_repo_conversation(db, repo.id)
    db.commit()


# 检查单仓库默认会话是否缺失并执行补建。
def _backfill_repo_conversation(db: Session, repo_id: uuid.UUID) -> None:
    conversation = ensure_default_conversation(db, repo_id)
    for model in [ChatMessage, ChatSession, AgentRun]:
        db.query(model).filter(model.repo_id == repo_id, model.conversation_id.is_(None)).update(
            {model.conversation_id: conversation.id},
            synchronize_session=False,
        )
    conversation.message_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).count()
