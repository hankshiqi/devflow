# 学习导读｜启动记忆 MCP 子进程，通过 stdio 交换 JSON-RPC 消息并读取工具结果。
# 初始化握手完成后才能调用工具；结束时清理进程和管道资源。
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings


# 表示 MCP 子进程启动、握手或工具调用失败。
class McpClientError(RuntimeError):
    pass


# 管理 MCP stdio 子进程、协议握手、请求 ID 和工具调用生命周期。
class StdioMcpClient:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self, command: str | None = None, args: list[str] | None = None, cwd: str | None = None) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        self.command = command or sys.executable
        self.args = args or [str(backend_root / "app" / "mcp" / "memory_server.py")]
        self.cwd = cwd or str(backend_root)
        self._next_id = 1

    # 启动/握手 MCP 服务后发送 tools/call，解析结果并在结束时清理子进程。
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        process = await self._start()
        try:
            await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            return await self._request(process, "tools/call", {"name": name, "arguments": arguments})
        finally:
            await self._stop(process)

    # 启动真实 MCP 子进程并完成初始化和 tools/list 握手。
    async def inspect_server(self) -> dict[str, Any]:
        """Perform a real MCP handshake and return the server's advertised tools."""
        process = await self._start()
        try:
            initialize = await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            tools = await self._request(process, "tools/list", {})
            return {"initialize": initialize, "tools": tools.get("tools") or []}
        finally:
            await self._stop(process)

    # 完成 MCP 协议初始化及通知，随后才发送正常工具请求。
    async def _initialize(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        return await self._request(
            process,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "devflow-web-agent", "version": "0.1.0"},
            },
        )

    # 创建 MCP 子进程及 stdin/stdout 管道，准备后续 JSON-RPC 通信。
    async def _start(self) -> asyncio.subprocess.Process:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["DATABASE_URL"] = settings.database_url
        env.setdefault("AUTO_SYNC_ENABLED", "false")
        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return process

    # 发送无需响应 ID 的 JSON-RPC 通知消息。
    async def _notify(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> None:
        await self._write(process, {"jsonrpc": "2.0", "method": method, "params": params})

    # 发送带 ID 的 JSON-RPC 请求并等待匹配响应，将协议错误转换为业务异常。
    async def _request(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        await self._write(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if process.stdout is None:
            raise McpClientError("MCP process has no stdout")
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        except TimeoutError as exc:
            raise McpClientError(f"MCP request timed out: {method}") from exc
        if not line:
            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace")
            raise McpClientError(f"MCP server closed stdout. {stderr}")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise McpClientError(f"MCP server returned an invalid JSON response for {method}") from exc
        if "error" in response:
            error = response["error"]
            raise McpClientError(str(error.get("message") or error))
        return response.get("result") or {}

    # 把 JSON-RPC 消息写入子进程 stdin，按行分隔并刷新管道。
    async def _write(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        if process.stdin is None:
            raise McpClientError("MCP process has no stdin")
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    # 关闭 MCP 子进程输入流，等待退出并在必要时终止残留进程。
    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            process.kill()
        await process.wait()
