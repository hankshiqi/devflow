# 学习导读｜所有业务 Agent 的公共接口，约定名称、工具声明、run 输入输出及错误格式。
# ABC/abstractmethod 要求具体子类实现 run；继承此类本身不会自动创建模型或调用工具。
from abc import ABC, abstractmethod
from typing import Any


# 定义所有 Agent 的共同调用约定；抽象 run 必须由子类实现。
class BaseAgent(ABC):
    name: str
    description: str
    available_tools: list[str]

    # 统一异步分析入口：input_data 是任务数据，context 提供仓库、证据和运行环境。
    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # 预留输出校验扩展点；当前默认原样返回，具体 Agent 可使用 Pydantic 进一步验证。
    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        return output

    # 把异常转换为统一错误字典，并仅附带定位所需的仓库信息。
    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "error": str(error),
            "agent": self.name,
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }
