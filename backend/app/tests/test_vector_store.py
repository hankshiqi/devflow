# 学习导读｜回归测试：覆盖 Milvus 连接/维度、过滤条件、写入错误和 SQL/向量存储边界。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import uuid

import pytest

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_knowledge_texts, validate_embedding_contract
from app.services.rag.vector_store import _clip_utf8, _collection_dimensions, upsert_milvus_documents


# 按字节截断需保留完整 Unicode 字符，验证多字节中文边界。
def test_clip_utf8_limits_encoded_length_without_breaking_characters() -> None:
    value = "\u732b" * 3000
    clipped = _clip_utf8(value, 8192)

    assert len(clipped.encode("utf-8")) <= 8192
    assert clipped
    assert set(clipped) == {"\u732b"}


# 集合维度需从实际 Schema 读取，而非只相信本地配置。
def test_collection_dimension_is_read_from_milvus_schema() -> None:
    assert _collection_dimensions(
        {"fields": [{"name": "embedding", "params": {"dim": "1024"}}]}
    ) == 1024


# 单仓库配置不能任意偏离当前全局向量空间契约。
def test_embedding_contract_rejects_per_repository_vector_spaces() -> None:
    with pytest.raises(ValueError, match="Embedding contract mismatch"):
        validate_embedding_contract("deterministic", "other-model", 384)


# 演示向量可以显式启用，但不能成为生产失败时的静默替代。
def test_explicit_deterministic_provider_is_allowed_without_becoming_a_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "deterministic")
    monkeypatch.setattr(settings, "embedding_model", "deterministic-local")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)

    vectors = embed_knowledge_texts(
        ["token failure"],
        provider="deterministic",
        model="deterministic-local",
        dimensions=8,
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == 8


# 真实模型失败应报错，不生成另一向量空间的哈希向量冒充成功。
def test_embedding_provider_failure_is_not_replaced_with_hash_vectors(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "openai_compatible")
    monkeypatch.setattr(settings, "embedding_model", "configured-model")
    monkeypatch.setattr(settings, "embedding_dimensions", 8)
    monkeypatch.setattr(
        "app.services.rag.embeddings._embed_openai_compatible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider offline")),
    )

    with pytest.raises(RuntimeError, match="provider offline"):
        embed_knowledge_texts(
            ["token failure"],
            provider="openai_compatible",
            model="configured-model",
            dimensions=8,
        )


# 维度不符时应在连接前拒绝，不能靠截断/补零凑维度。
def test_milvus_upsert_rejects_dimension_fitting_before_connecting(monkeypatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimensions", 4)
    document = Document(
        id=uuid.uuid4(),
        repo_id=uuid.uuid4(),
        source_type="issue",
        source_id=uuid.uuid4(),
        title="Token failure",
        content="Refresh failed.",
        meta={},
    )

    with pytest.raises(ValueError, match="Refusing to truncate or pad"):
        upsert_milvus_documents([document], [[1.0, 0.0]])
