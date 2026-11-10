# 学习导读｜统一初始化 Python 日志输出级别和格式，便于追踪后台任务及外部服务错误。
import logging


# 根据配置统一设置日志级别、格式和日志库，避免各模块各自初始化。
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
