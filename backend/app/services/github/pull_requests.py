# 学习导读｜封装 GitHub PR、变更文件和行级 Review 评论接口。
# 文件接口返回的 patch 是增删行文本；同步服务将它保存到 PRFile.patch。
from app.services.github.client import GitHubClient


# 请求 GitHub 仓库 PR 列表，供同步核心处理。
async def list_pull_requests(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls", {"state": "all", "per_page": limit})


# 请求 PR 变更文件列表；返回文件增删统计和可用的 patch 字段，并非读取整个源文件。
async def list_pull_request_files(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/files", {"per_page": 100})


# 请求 PR 的行级 Review 评论，保留评论路径和行号供审查使用。
async def list_pull_request_review_comments(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/comments", {"per_page": 100})
