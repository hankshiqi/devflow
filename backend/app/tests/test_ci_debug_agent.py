# 学习导读｜回归测试：覆盖 ci / debug / agent。
# 阅读方式：先看 fixture/替身如何构造输入，再看待测函数调用与 assert 约束。
# 测试中的临时库、模拟模型和示例向量仅用于验证行为，不代表生产环境配置。
import pytest

from app.services.agents.ci_debug_agent import CIDebugAgent


# CI 诊断输出应覆盖实际失败 Job 的上下文，而非只看运行标题。
@pytest.mark.asyncio
async def test_ci_debug_agent_includes_failed_job_context() -> None:
    result = await CIDebugAgent().run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    assert result["failure_type"] == "test"
    assert "pytest" in result["failure_summary"]
