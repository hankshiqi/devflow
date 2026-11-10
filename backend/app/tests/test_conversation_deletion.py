# 学习导读｜回归测试：覆盖 conversation / deletion。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    Base,
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
from app.services.conversations import delete_conversation


# 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# 删除会话需清理其封存记忆和工作流依赖，避免外键残留。
def test_delete_conversation_removes_session_memory_and_workflow_dependencies() -> None:
    db = _db_session()
    repo = Repository(owner="devflow", name="conversation-delete", full_name=f"devflow/{uuid.uuid4().hex}")
    db.add(repo)
    db.flush()
    target = Conversation(repo_id=repo.id, title="hi，你好")
    replacement = Conversation(repo_id=repo.id, title="保留会话")
    db.add_all([target, replacement])
    db.flush()

    run = AgentRun(
        repo_id=repo.id,
        conversation_id=target.id,
        user_message="hi，你好",
        route="test",
        tool_calls=[],
        final_answer="你好",
    )
    user_message = ChatMessage(repo_id=repo.id, conversation_id=target.id, role="user", content="hi，你好")
    assistant_message = ChatMessage(repo_id=repo.id, conversation_id=target.id, role="assistant", content="你好")
    session = ChatSession(repo_id=repo.id, conversation_id=target.id, route="test", transcript_json={}, digest="你好")
    workflow = AgentWorkflowRun(repo_id=repo.id, conversation_id=target.id, goal="test")
    db.add_all([run, user_message, assistant_message, session, workflow])
    db.flush()

    db.add_all(
        [
            ChatFeedback(
                repo_id=repo.id,
                conversation_id=target.id,
                assistant_message_id=assistant_message.id,
                run_id=run.id,
                rating="helpful",
            ),
            MemoryCandidate(
                repo_id=repo.id,
                conversation_id=target.id,
                session_id=session.id,
                kind="fact",
                title="candidate",
                content="must be deleted before the session",
            ),
            RecallEvent(repo_id=repo.id, conversation_id=target.id, tool_name="memory.search"),
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id="task-1",
                agent_name="test-agent",
                task_type="test",
            ),
            ConversationMemory(repo_id=repo.id, conversation_id=target.id),
            EvidenceItem(
                repo_id=repo.id,
                conversation_id=target.id,
                source_type="test",
                source_id="conversation-delete",
                title="evidence",
                content="delete regression",
            ),
        ]
    )
    db.commit()

    active = delete_conversation(db, repo.id, target.id)
    db.commit()

    assert active.id == replacement.id
    assert db.get(Conversation, target.id).status == "deleted"
    for model in [
        ChatFeedback,
        MemoryCandidate,
        RecallEvent,
        AgentTaskRun,
        AgentWorkflowRun,
        ChatMessage,
        ChatSession,
        AgentRun,
        ConversationMemory,
        EvidenceItem,
    ]:
        assert db.query(model).count() == 0
