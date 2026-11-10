# 学习导读｜回归测试：覆盖 GitHub 返回值到 Issue/PR/CI ORM 对象的同步。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Issue, Repository
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository


# 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# GitHub 实体同步应保存事实，但不能隐式触发 Embedding/Milvus。
@pytest.mark.asyncio
async def test_sync_repository_persists_facts_without_triggering_rag(monkeypatch) -> None:
    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def fake_list_issues(*_args, **_kwargs):
        return [
            {
                "id": 7800,
                "number": 78,
                "title": "Error: Could not import 'app'.",
                "body": "Realistic issue body",
                "state": "open",
                "labels": [{"name": "bug"}],
                "user": {"login": "contributor"},
                "assignees": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "closed_at": None,
            }
        ]

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        raise AssertionError("GitHub fact sync must not depend on embeddings or Milvus")

    monkeypatch.setattr("app.services.repos_sync.list_issues", fake_list_issues)
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)

    db = _db_session()
    repo = Repository(
        owner="openai",
        name="openai-quickstart-python",
        full_name="openai/openai-quickstart-python",
        provider="github",
        api_base_url="https://api.github.com",
    )
    db.add(repo)
    db.commit()

    result = await sync_repository(
        db,
        repo,
        RepoSyncRequest(sync_pull_requests=False, sync_workflow_runs=False, limit=1),
    )

    issue = db.query(Issue).filter(Issue.repo_id == repo.id).one()
    assert result.status == "completed"
    assert result.synced == {"issues": 1, "pull_requests": 0, "workflow_runs": 0}
    assert issue.number == 78
    assert issue.title == "Error: Could not import 'app'."
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 0
