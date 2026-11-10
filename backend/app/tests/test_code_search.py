# 学习导读｜回归测试：覆盖 code / search。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import uuid
from types import SimpleNamespace

from app.services.code_search import query_terms, search_checkout_code, search_repository_code


# 查询词提取应保留函数名、路径等代码线索，过滤泛化停用词。
def test_query_terms_prioritizes_code_identifiers() -> None:
    terms = query_terms("为什么 refreshAccessToken 在 auth/service.py 里失败？")

    assert "refreshaccesstoken" in terms
    assert "auth/service.py" in terms


# 工作区搜索直接读磁盘，不依赖预先建立数据库索引。
def test_search_checkout_code_reads_current_files_without_database_index(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "auth.py"
    source.write_text(
        "def refresh_access_token():\n"
        "    return rotate_credentials()\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("REFRESH_ACCESS_TOKEN=secret\n", encoding="utf-8")

    results = search_checkout_code(tmp_path, "refresh_access_token", limit=5)

    assert results
    assert results[0]["source_type"] == "workspace_file"
    assert results[0]["source_id"] == "src/auth.py"
    assert results[0]["metadata"]["retrieval_mode"] == "workspace"
    assert all(item["source_id"] != ".env" for item in results)


# 指定子目录时只搜索该范围，避免返回工作树其他区域。
def test_search_checkout_code_respects_worktree_subdirectory(tmp_path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: backend-tests\n", encoding="utf-8")
    (tmp_path / "src" / "test.py").write_text("backend_tests = True\n", encoding="utf-8")

    results = search_checkout_code(
        tmp_path,
        "backend-tests",
        relative_path=".github/workflows",
    )

    assert results and {item["source_id"] for item in results} == {".github/workflows/test.yml"}


# 仓库级搜索应优先使用配置的本地 checkout。
def test_search_repository_code_uses_configured_local_checkout(tmp_path) -> None:
    (tmp_path / "service.py").write_text("def handle_payment_webhook():\n    pass\n", encoding="utf-8")
    repo = SimpleNamespace(id=uuid.uuid4(), local_path=str(tmp_path), full_name="local/demo")

    results = search_repository_code(repo, "handle_payment_webhook")

    assert results and results[0]["metadata"]["repo_id"] == str(repo.id)


# 直接修改磁盘文件后无需重建向量索引，搜索即可看到新内容。
def test_search_checkout_code_observes_file_changes_without_reindex(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def old_handler():\n    pass\n", encoding="utf-8")
    assert search_checkout_code(tmp_path, "old_handler")

    source.write_text("def current_handler():\n    pass\n", encoding="utf-8")

    assert search_checkout_code(tmp_path, "current_handler")
    assert search_checkout_code(tmp_path, "old_handler") == []
