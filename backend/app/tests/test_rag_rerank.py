# 学习导读｜回归测试：覆盖 rag / rerank。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
from app.services.rag.rerank import rerank_documents


# 重排应优先保留路径及查询词明确匹配的候选。
def test_rerank_prefers_exact_path_and_query_terms() -> None:
    candidates = [
        {
            "id": "general",
            "title": "README setup",
            "snippet": "General local development notes",
            "score": 0.8,
            "vector_score": 0.8,
            "keyword_score": 0.1,
            "metadata": {},
        },
        {
            "id": "auth",
            "title": "backend/app/auth/middleware.py#1",
            "snippet": "validate token and refresh session when login expires",
            "score": 0.5,
            "vector_score": 0.5,
            "keyword_score": 1.0,
            "metadata": {"path": "backend/app/auth/middleware.py"},
        },
    ]

    results = rerank_documents("login token auth middleware", candidates, limit=2)

    assert results[0]["id"] == "auth"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert "rank_reason" in results[0]
