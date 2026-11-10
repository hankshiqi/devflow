# 学习导读｜封装 GitHub Issue 列表接口，供仓库同步服务调用。
from app.services.github.client import GitHubClient


# 请求远端 GitHub Issues 列表，供同步核心转换为 ORM 数据。
async def list_issues(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    items = await client.get(f"/repos/{owner}/{repo}/issues", {"state": "all", "per_page": limit})
    return [item for item in items if "pull_request" not in item]
