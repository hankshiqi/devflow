# 学习导读｜接口数据契约：周报请求及汇总。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import date
from uuid import UUID

from pydantic import BaseModel


# WeeklyReportRequest：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、start_date、end_date。
class WeeklyReportRequest(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    start_date: date
    end_date: date


# WeeklyReportResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：report_markdown。
class WeeklyReportResponse(BaseModel):
    report_markdown: str
