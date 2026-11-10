# 学习导读｜统一 GitHub HTTP 客户端，集中处理 API 地址、认证请求头和 HTTP 错误。
# GET 用于读取；POST/PATCH 用于远程写入，具体业务动作由上层控制。
from typing import Any

import httpx

from app.core.config import settings


# 集中封装 GitHub API 的地址、认证和 HTTP 方法；业务对象转换由上游负责。
class GitHubClient:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = token or settings.github_token
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")

    # 生成访问 GitHub 所需的请求头，并按是否配置 Token 添加认证信息。
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # 执行 JSON GET 请求，HTTP 错误向上抛出，由业务入口决定如何处理。
    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    # 读取日志归档等二进制响应，避免把 ZIP 内容误当 JSON 解析。
    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.content

    # 执行 GitHub 创建/提交类请求，调用方需要保证动作已获授权。
    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    # 执行 GitHub 更新类请求，与 PR 文件字段 patch 的含义不同。
    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.patch(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    # 读取仓库基本信息，用于连接时校验仓库和填充元数据。
    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        return await self.get(f"/repos/{owner}/{repo}")

    # 向指定 Issue/PR 创建评论，属于远程写操作。
    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    # 在仓库中创建 Issue，属于远程写操作。
    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    # 更新 Issue 的 state 为 closed，属于远程写操作。
    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    # 为 Issue 添加标签，属于远程写操作。
    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})
