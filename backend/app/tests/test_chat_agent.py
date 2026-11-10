# 学习导读｜回归测试：覆盖 模型与工具循环、重复调用保护、记忆读取、流式事件和上下文预算。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.api.routes.chat import _build_conversation_tools, _repo_checkout_path
from app.core.config import settings
from app.services.agents.chat_agent import ChatAgent


class ScriptedToolModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "ScriptedToolModel":
        self.tools = tools
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__list_files", "args": {"path": ".", "limit": 5}, "id": "call_list"},
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": "call_read"},
                ],
            )
        return AIMessage(content="README.md says this is the project overview.")


class SummaryModel:
    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "SummaryModel":
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content="The README says this project coordinates AI agents.")


class RecentRecallModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0
        self.bind_tools_calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "RecentRecallModel":
        self.bind_tools_calls += 1
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(content="你刚刚说的是：hi")


class CapturingModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.messages: list[Any] = []

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "CapturingModel":
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.messages = messages
        return AIMessage(content="No team members are configured.")


class SingleToolModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "SingleToolModel":
        self.tools = tools
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": "call_read"},
                ],
            )
        return AIMessage(content="The README has been read.")


class RepeatingToolModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "RepeatingToolModel":
        self.tools = tools
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "workspace__read_file", "args": {"path": "README.md"}, "id": f"call_read_{self.calls}"},
            ],
        )


class RewrittenRagModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "RewrittenRagModel":
        self.tools = tools
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag__search_similar_documents",
                        "args": {"query": "token refresh timeout history"},
                        "id": "call_rag_history",
                    }
                ],
            )
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag__search_similar_documents",
                        "args": {"query": "idempotent retry decision"},
                        "id": "call_rag_decision",
                    }
                ],
            )
        return AIMessage(content="The history and retry decision agree: retry once with an idempotency key.")


class UsageModel:
    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "UsageModel":
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(
            content="usage captured",
            usage_metadata={"input_tokens": 123, "output_tokens": 7, "total_tokens": 130},
        )


class OverflowThenSuccessModel:
    # 初始化本对象持有的依赖与配置；是否连接外部服务取决于下面实际调用的构造逻辑。
    def __init__(self) -> None:
        self.calls = 0

    # 测试桩方法：接收并保存 Agent 提供的工具集合，再返回模型替身自身。
    def bind_tools(self, tools: list[Any]) -> "OverflowThenSuccessModel":
        return self

    # 测试桩方法：返回预置模型/评分结果，不访问外部服务；具体场景由所在测试函数决定。
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("maximum context length exceeded")
        return AIMessage(content="recovered after compact projection")


# 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
def _history_context(*, complete_and_exact: bool, exact_suffix_messages: int = 3) -> dict[str, Any]:
    return {
        "server_messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user", "content": "我刚刚说了什么？"},
        ],
        "compression_stats": {
            "history_coverage": {
                "complete_and_exact": complete_and_exact,
                "exact_suffix_messages": exact_suffix_messages,
                "omitted_messages": 0 if complete_and_exact else 1,
                "truncated_messages": 0,
                "source_limit_reached": not complete_and_exact,
                "has_compaction_boundary": False,
                "microcompacted_messages": 0,
            }
        },
    }


# 验证模型返回用量和提示预算被分别记录，便于区分计费信息与本地估算。
@pytest.mark.asyncio
async def test_chat_agent_records_provider_usage_and_prompt_breakdown() -> None:
    final = await ChatAgent(model=UsageModel()).run(  # type: ignore[arg-type]
        {"message": "Give me a concise answer."},
        {"tools": {}},
    )

    assert final["answer"] == "usage captured"
    assert final["model_usage"][-1]["input_tokens"] == 123
    assert final["model_usage"][-1]["source"] == "provider"
    prompt_call = final["context_diagnostics"]["prompt_calls"][-1]
    assert prompt_call["estimated_prompt_tokens"] > 0
    assert prompt_call["tool_schema_tokens"] == 0


# 验证上下文溢出只重试失败的模型请求，不重跑已经完成的工具。
@pytest.mark.asyncio
async def test_context_overflow_retries_only_the_failed_model_call() -> None:
    model = OverflowThenSuccessModel()
    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "Continue the task."},
        {"tools": {}},
    )

    assert final["answer"] == "recovered after compact projection"
    assert model.calls == 2
    assert final["context_diagnostics"]["reactive_overflow_retry"] is True
    assert any(event["type"] == "context_overflow_recovery" for event in final["agent_events"])


# 验证模型输入可以压缩旧观察，但原始审计轨迹保持完整。
def test_runtime_tool_projection_compacts_old_observations_without_mutating_audit_state() -> None:
    agent = ChatAgent(model=None)
    state = {
        "observations": [
            {
                "tool_name": f"tool-{index}",
                "arguments": {"query": "auth"},
                "observation": {"answer_preview": (f"result-{index} " * 5000)},
            }
            for index in range(4)
        ],
        "agent_events": [],
    }
    original = state["observations"][0]["observation"]["answer_preview"]

    prompt_observations, _events, stats = agent._runtime_tool_context(state)  # type: ignore[arg-type]

    assert stats["estimated_tokens_saved"] > 0
    assert stats["prompt_observation_tokens"] <= stats["observation_budget_tokens"]
    assert state["observations"][0]["observation"]["answer_preview"] == original
    assert len(prompt_observations) >= 1


# 近期历史已覆盖问题时，应直接使用已注入消息，不额外查询记忆。
@pytest.mark.asyncio
async def test_recent_recall_uses_injected_history_without_memory_tool() -> None:
    model = RecentRecallModel()
    memory_tool_calls = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal memory_tool_calls
        memory_tool_calls += 1
        return {"route": "thread_context", "tool_calls": [], "answer": "user: hi", "citations": []}

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "我刚刚说了什么？"},
        {
            **_history_context(complete_and_exact=True),
            "tools": {"devflow_get_thread_context": get_thread_context},
        },
    )

    assert final["answer"] == "你刚刚说的是：hi"
    assert final["route"] == "direct_answer"
    assert final["tool_calls"] == []
    assert memory_tool_calls == 0
    assert model.calls == 1
    assert model.bind_tools_calls == 0
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["mode"] == "direct_recent_history"
    assert policy["reason"] == "exact_recent_messages_available"


# 当前提示中的历史不完整时，应通过记忆工具补取线程上下文。
@pytest.mark.asyncio
async def test_recent_recall_fetches_thread_context_when_prompt_history_is_incomplete() -> None:
    model = RecentRecallModel()
    memory_tool_calls = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal memory_tool_calls
        memory_tool_calls += 1
        return {
            "route": "thread_context",
            "tool_calls": [
                {"tool_name": "devflow_get_thread_context", "arguments": input_data, "status": "success"}
            ],
            "answer": "user: hi",
            "citations": [],
        }

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "我刚刚说了什么？"},
        {
            **_history_context(complete_and_exact=False, exact_suffix_messages=1),
            "tools": {"devflow_get_thread_context": get_thread_context},
        },
    )

    assert memory_tool_calls == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_get_thread_context"]
    assert not any(call["tool_name"] == "devflow_search_evidence" for call in final["tool_calls"])
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "recent_history_incomplete"


# 精确回忆原话时需核对线程上下文，不能凭语义相似资料作答。
@pytest.mark.asyncio
async def test_exact_recall_request_verifies_with_thread_context() -> None:
    invocations = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def get_thread_context(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "thread_context",
            "tool_calls": [
                {"tool_name": "devflow_get_thread_context", "arguments": input_data, "status": "success"}
            ],
            "answer": "user: hi",
            "citations": [],
        }

    context = _history_context(complete_and_exact=True)
    context["server_messages"][-1]["content"] = "请逐字核实我刚刚说的原话"
    final = await ChatAgent(model=RecentRecallModel()).run(  # type: ignore[arg-type]
        {"message": "请逐字核实我刚刚说的原话"},
        {**context, "tools": {"devflow_get_thread_context": get_thread_context}},
    )

    assert invocations == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_get_thread_context"]
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "exact_verification_requested"


# 跨封存会话读取原话时，应调用原始会话事件工具。
@pytest.mark.asyncio
async def test_cross_session_recall_uses_session_events() -> None:
    invocations = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_session_events(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "session_events",
            "tool_calls": [
                {"tool_name": "devflow_read_session_events", "arguments": input_data, "status": "success"}
            ],
            "answer": "上次会话决定使用短期令牌。",
            "citations": [],
        }

    context = _history_context(complete_and_exact=True)
    context["server_messages"][-1]["content"] = "上次会话我们决定了什么？"
    final = await ChatAgent(model=RecentRecallModel()).run(  # type: ignore[arg-type]
        {"message": "上次会话我们决定了什么？"},
        {**context, "tools": {"devflow_read_session_events": read_session_events}},
    )

    assert invocations == 1
    assert [call["tool_name"] for call in final["tool_calls"]] == ["devflow_read_session_events"]
    policy = next(event for event in final["agent_events"] if event["type"] == "context_policy")
    assert policy["reason"] == "cross_session_request"


# 验证 Agent 可以先列目录，再根据结果继续读文件，最后生成回答。
@pytest.mark.asyncio
async def test_chat_agent_can_continue_from_list_files_to_read_file() -> None:
    model = ScriptedToolModel()
    tool_invocations: list[tuple[str, dict[str, Any]]] = []

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("list", input_data))
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md",
            "citations": [],
        }

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("read", input_data))
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [{"type": "workspace_file", "id": "README.md", "title": "README.md"}],
        }

    agent = ChatAgent(model=model)  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Find the overview files, then read README.md."},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert model.calls == 3
    assert [name for name, _input_data in tool_invocations] == ["list", "read"]
    assert final["route"] == "workspace_read_file"
    assert [call["tool_name"] for call in final["tool_calls"]] == ["workspace.list_files", "workspace.read_file"]
    assert [event["type"] for event in final["agent_events"] if event["type"] in {"tool_use", "tool_result"}] == [
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
    ]


# 验证工具尚未完成时前端已能收到开始事件，而不是结果出来后才补发全部过程。
@pytest.mark.asyncio
async def test_chat_agent_streams_tool_use_before_tool_finishes() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await release.wait()
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [{"type": "workspace_file", "id": "README.md", "title": "README.md"}],
        }

    agent = ChatAgent(model=SingleToolModel())  # type: ignore[arg-type]
    stream = agent.run_stream(
        {"message": "Read README.md."},
        {"tools": {"workspace.read_file": read_file}},
    )

    first_event = await stream.__anext__()
    assert first_event["event"] == "tool_use"
    assert first_event["data"]["tool_name"] == "workspace.read_file"
    assert not started.is_set()

    next_event = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert not next_event.done()
    release.set()
    tool_result = await asyncio.wait_for(next_event, timeout=1)
    assert tool_result["event"] == "tool_result"
    remaining_events = [event async for event in stream]
    assert remaining_events[-1]["event"] == "final"


# 项目介绍问题应主动预取 README，不完全依赖模型猜测需要哪个文件。
@pytest.mark.asyncio
async def test_project_overview_prefetches_readme_without_waiting_for_model_tool_choice() -> None:
    tool_invocations: list[tuple[str, dict[str, Any]]] = []

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("list", input_data))
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md\nREADME.zh-CN.md\npackage.json",
            "citations": [],
        }

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        tool_invocations.append(("read", input_data))
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": f"{input_data['path']}:1-1\n1: # Clowder AI",
            "citations": [{"type": "workspace_file", "id": input_data["path"], "title": input_data["path"]}],
        }

    agent = ChatAgent(model=SummaryModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "What is this project?"},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert [name for name, _input_data in tool_invocations] == ["list", "read", "read", "read"]
    assert final["route"] == "workspace_read_file"
    assert "coordinates AI agents" in final["answer"]
    assert len([event for event in final["agent_events"] if event["type"] == "tool_use"]) == 4
    assert len([event for event in final["agent_events"] if event["type"] == "tool_result"]) == 4


# 验证预检索证据能与 Agent 改写 query 后的多次检索共同使用。
@pytest.mark.asyncio
async def test_chat_agent_combines_prefetched_evidence_with_rewritten_rag_searches() -> None:
    model = RewrittenRagModel()
    queries: list[str] = []

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def search_rag(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or "")
        queries.append(query)
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"query": query},
                    "status": "success",
                }
            ],
            "answer": f"Evidence for {query}",
            "citations": [{"type": "project_doc", "id": query, "title": query}],
        }

    events = [
        event
        async for event in ChatAgent(model=model).run_stream(  # type: ignore[arg-type]
            {"message": "Why does token refresh fail, and what retry policy did we choose?"},
            {
                "tools": {"rag.search_similar_documents": search_rag},
                "evidence": [{"title": "prefetched auth overview"}],
                "server_context": "[Related Evidence]\nauth overview\n[/Related Evidence]",
            },
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert queries == ["token refresh timeout history", "idempotent retry decision"]
    assert [call["arguments"]["query"] for call in final["tool_calls"]] == queries
    strategy = next(event for event in final["agent_events"] if event["type"] == "rag_strategy")
    assert strategy["mode"] == "hybrid_rag"
    assert strategy["pre_retrieval"]["evidence_count"] == 1
    assert strategy["pre_retrieval"]["query"] == "Why does token refresh fail, and what retry policy did we choose?"
    assert strategy["pre_retrieval"]["sources"] == [
        {
            "source_id": "",
            "source_type": "evidence",
            "title": "prefetched auth overview",
            "score": None,
        }
    ]
    assert strategy["reasoning_retrieval"]["supports_query_rewrite"] is True
    assert final["completion"]["tool_call_count"] == 2


# 评测必须同时保留预检索和工具二次检索证据，避免少统计实际上下文。
@pytest.mark.asyncio
async def test_chat_agent_evaluation_mode_preserves_prefetch_and_tool_retrieval_contexts() -> None:
    model = RewrittenRagModel()

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def search_rag(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or "")
        source_id = f"source-{len(query)}"
        snippet = f"Evidence returned for {query}."
        return {
            "route": "semantic_search",
            "tool_calls": [{"tool_name": "rag.search_similar_documents", "arguments": {"query": query}, "status": "success"}],
            "answer": snippet,
            "citations": [{"type": "project_doc", "id": source_id, "title": query}],
            "retrieved_contexts": [snippet],
            "retrieval_results": [
                {
                    "id": source_id,
                    "source_id": source_id,
                    "source_type": "project_doc",
                    "title": query,
                    "snippet": snippet,
                    "score": 0.9,
                }
            ],
        }

    final = await ChatAgent(model=model).run(  # type: ignore[arg-type]
        {"message": "How should auth retries work?"},
        {
            "tools": {"rag.search_similar_documents": search_rag},
            "evaluation_mode": True,
            "evidence": [
                {
                    "source_id": "prefetch-1",
                    "source_type": "project_doc",
                    "title": "Authentication policy",
                    "snippet": "Authentication retries must be idempotent.",
                    "score": 1.0,
                }
            ],
        },
    )

    trace = final["evaluation_trace"]
    assert trace["target"] == "ChatAgent"
    assert [item["stage"] for item in trace["retrievals"]] == ["prefetch", "agent_tool", "agent_tool"]
    assert (
        "[project_doc] Authentication policy: Authentication retries must be idempotent."
        in trace["retrieved_contexts"]
    )
    assert any(
        item.endswith("Evidence returned for token refresh timeout history.")
        for item in trace["retrieved_contexts"]
    )
    assert len(trace["tool_observations"]) == 2

    normal = await ChatAgent(model=SummaryModel()).run(  # type: ignore[arg-type]
        {"message": "Summarize the evidence."},
        {"tools": {}, "evidence": []},
    )
    assert "evaluation_trace" not in normal


# 没有可用模型时，项目介绍兜底仍应使用真实工作区证据。
@pytest.mark.asyncio
async def test_project_overview_without_llm_uses_workspace_evidence(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def list_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": input_data, "status": "success"}],
            "answer": "README.md frontend/package.json",
            "citations": [],
        }

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-2\n1: # DevFlow AI\n2: Core capability: repository analysis.",
            "citations": [],
        }

    events = [
        event
        async for event in ChatAgent(model=None).run_stream(
            {"message": "What does this project do?"},
            {"tools": {"workspace.list_files": list_files, "workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert "Core capability: repository analysis" in final["answer"]
    assert final["completion"]["tool_call_count"] == 2
    assert final["completion"]["tool_error_count"] == 0
    assert final["completion"]["status"] == "completed"
    assert final["completion"]["reason"] == "project_context_loaded"


# 没有模型时，Issue 问题应使用已检索 Issue 证据，不误调用无关源码工具。
@pytest.mark.asyncio
async def test_open_issue_question_without_llm_uses_prefetched_issue_evidence(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def unexpected_workspace_call(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Issue query must not be routed to workspace tools")

    events = [
        event
        async for event in ChatAgent(model=None).run_stream(
            {"message": "请列出当前仓库未关闭的 Issue，并说明你使用了哪些数据。"},
            {
                "tools": {"workspace.list_files": unexpected_workspace_call},
                "evidence": [
                    {
                        "source_type": "issue",
                        "source_id": "issue-18",
                        "title": "Issue #18: Token refresh race",
                        "snippet": "Token refresh can race under concurrent requests. state: open",
                        "metadata": {"number": 18, "state": "open"},
                    },
                    {
                        "source_type": "issue",
                        "source_id": "issue-11",
                        "title": "Issue #11: Retired task",
                        "snippet": "This task is complete. state: closed",
                        "metadata": {"number": 11, "state": "closed"},
                    },
                ],
            },
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert "Issue #18: Token refresh race" in final["answer"]
    assert "Issue #11: Retired task" not in final["answer"]
    assert "本轮没有调用工作区源码工具" in final["answer"]
    assert final["tool_calls"] == []
    assert final["route"] == "prefetched_evidence"
    assert final["completion"]["status"] == "completed"
    assert final["completion"]["reason"] == "prefetched_evidence_available"


# 检索工具应使用模型改写后的 query，而非无条件覆盖成最初用户问题。
@pytest.mark.asyncio
async def test_conversation_rag_tool_prefers_agent_rewritten_query(monkeypatch) -> None:
    repo = SimpleNamespace(id=uuid.uuid4(), full_name="acme/auth-service")
    conversation = SimpleNamespace(id=uuid.uuid4())
    captured: dict[str, Any] = {}

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    def fake_semantic_search(repo_arg, query, context, db):
        captured["search_query"] = query
        return (
            "rewritten evidence",
            [{"type": "project_doc", "id": "doc-1", "title": "Retry policy"}],
            [{"source_type": "project_doc", "source_id": "doc-1", "title": "Retry policy"}],
        )

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    def fake_record_recall_event(*args, **kwargs):
        captured["recall_query"] = kwargs["query"]
        captured["metadata"] = kwargs["metadata"]

    monkeypatch.setattr("app.api.routes.chat._semantic_search", fake_semantic_search)
    monkeypatch.setattr("app.api.routes.chat.record_recall_event", fake_record_recall_event)
    tools = _build_conversation_tools(repo, conversation, object())

    result = await tools["rag.search_similar_documents"](
        {
            "message": "Why does authentication fail?",
            "query": "refresh token idempotency race condition",
        },
        {},
    )

    assert captured["search_query"] == "refresh token idempotency race condition"
    assert captured["recall_query"] == "refresh token idempotency race condition"
    assert captured["metadata"]["query_rewritten"] is True
    assert result["tool_calls"][0]["arguments"]["query"] == "refresh token idempotency race condition"


# 同一仓库已存在 checkout 时应复用，避免重复生成不同路径缓存。
def test_repo_checkout_path_reuses_existing_same_full_name_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    stale_checkout = tmp_path / "old-id-296569015__clowder-ai"
    stale_checkout.mkdir()
    repo = SimpleNamespace(id=uuid.uuid4(), full_name="296569015/clowder-ai")

    assert _repo_checkout_path(repo) == stale_checkout.resolve()


# 空团队明确表示没有负责人候选，不能鼓励模型自行编造。
def test_assignment_prompt_treats_empty_team_as_no_candidates() -> None:
    agent = ChatAgent(model=SummaryModel())  # type: ignore[arg-type]
    prompt = agent._tool_calling_user_prompt(
        {
            "message": "这个桌面化 Issue 应该分给谁？",
            "context": {"team_members": [], "server_context": "README mentions 布偶猫 / opus as project personas."},
            "observations": [],
            "agent_events": [],
            "tools": {},
        }
    )

    assert "team_members: []" in prompt
    assert "当前仓库还没有配置 DevFlow 团队成员" in prompt
    assert "不要把仓库内容" in prompt


# 最终回答阶段仍需保留团队候选范围，防止前面校验后又编出新成员。
@pytest.mark.asyncio
async def test_final_prompt_keeps_assignment_candidates_scoped_to_team_members() -> None:
    model = CapturingModel()
    agent = ChatAgent(model=model)  # type: ignore[arg-type]
    await agent._final_from_observations(
        {
            "message": "这个桌面化 Issue 应该分给谁？",
            "context": {"team_members": [], "server_context": "README mentions 布偶猫 / opus as project personas."},
            "observations": [{"answer": "Issue text mentions 布偶猫 and desktop work."}],
            "agent_events": [],
        }
    )
    prompt = "\n".join(str(message.content) for message in model.messages)

    assert "team_members: []" in prompt
    assert "团队比喻都不能当成人" in prompt
    assert "在团队面板中添加开发者" in prompt


# 验证临时工具错误可有限重试，不无限循环。
@pytest.mark.asyncio
async def test_chat_agent_retries_transient_tool_error_once() -> None:
    attempts = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def flaky_read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [],
        }

    agent = ChatAgent(model=SingleToolModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Read README.md."},
            {"tools": {"workspace.read_file": flaky_read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert attempts == 2
    assert final["tool_calls"][0]["status"] == "success"
    assert any(event["type"] == "tool_retry" and event["error_kind"] == "timeout" for event in final["agent_events"])
    assert final["completion"]["status"] == "completed"


# 验证连续相同工具/参数触发循环保护，避免无进展地消耗预算。
@pytest.mark.asyncio
async def test_chat_agent_stops_repeated_identical_tool_calls() -> None:
    invocations = 0

    # 测试辅助函数：为所在测试构造隔离数据或替身；具体验证目标见外层测试。
    async def read_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        nonlocal invocations
        invocations += 1
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": input_data, "status": "success"}],
            "answer": "README.md:1-1\n1: # Project",
            "citations": [],
        }

    agent = ChatAgent(model=RepeatingToolModel())  # type: ignore[arg-type]
    events = [
        event
        async for event in agent.run_stream(
            {"message": "Read README.md."},
            {"tools": {"workspace.read_file": read_file}},
        )
    ]

    final = [event for event in events if event["event"] == "final"][-1]["data"]
    assert invocations == agent.max_repeated_tool_calls
    assert final["route"] == "tool_loop_guard"
    assert final["completion"]["reason"] == "repeated_tool_call"
    assert any(event["type"] == "agent_loop_stop" and event["reason"] == "repeated_tool_call" for event in final["agent_events"])
