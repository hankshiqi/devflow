# 学习导读｜回归测试：覆盖 code / indexing。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import subprocess
import uuid
from types import SimpleNamespace

from app.core.config import settings
from app.services import code_analysis


# 远程更新失败时验证可继续使用已有 checkout，并明确这是缓存。
def test_sync_checkout_uses_cached_checkout_when_update_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
    )
    checkout = code_analysis._repo_checkout_path(repo)
    (checkout / ".git").mkdir(parents=True)

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    def fail_run(args, cwd=None):
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="network failed")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == checkout


# local 模式应直接使用配置目录，不自动 fetch/checkout 修改用户工作树。
def test_sync_checkout_uses_configured_local_repo_without_updating(tmp_path, monkeypatch) -> None:
    local_repo = tmp_path / "local-repo"
    local_repo.mkdir()
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(local_repo),
        checkout_mode="local",
    )

    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: path == local_repo.resolve())

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    def fail_run(args, cwd=None):
        raise AssertionError("local repository indexing should not fetch, checkout, or pull")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == local_repo.resolve()


# 受管克隆目标已被其他文件占用时应拒绝，避免覆盖既有内容。
def test_sync_checkout_rejects_non_empty_configured_download_target(tmp_path, monkeypatch) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "README.md").write_text("# occupied\n", encoding="utf-8")
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(target),
        checkout_mode="managed",
    )

    monkeypatch.setattr(code_analysis.shutil, "which", lambda name: "git")
    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: False)

    try:
        code_analysis._sync_checkout(repo, None)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected configured non-empty download target to be rejected")
