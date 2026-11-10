# 学习导读｜接口数据契约：消息反馈、审核与统计。
# Pydantic 模型负责请求验证/响应序列化，不负责创建数据库表；持久化模型见 db/models.py。
# 类型后的 | None 表示允许空值；字段是否可以省略还取决于是否定义了默认值。
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


FeedbackRating = Literal["helpful", "unhelpful"]
FeedbackReason = Literal[
    "inaccurate",
    "not_relevant",
    "missing_context",
    "unreliable_citation",
    "tool_error",
    "other",
]
ReviewStatus = Literal["open", "in_review", "resolved", "dismissed"]


# ChatFeedbackCreate：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：repo_id、conversation_id、assistant_message_id、rating。
class ChatFeedbackCreate(BaseModel):
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: str | None = Field(default=None, max_length=1000)

    # 校验负面反馈必须包含原因，避免无说明的无效反馈记录。
    @model_validator(mode="after")
    def validate_reason(self) -> "ChatFeedbackCreate":
        if self.rating == "helpful" and self.reason is not None:
            raise ValueError("有帮助反馈不需要填写负反馈原因")
        return self


# ChatFeedbackReview：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：review_status。
class ChatFeedbackReview(BaseModel):
    review_status: ReviewStatus
    review_note: str | None = Field(default=None, max_length=2000)


# ChatFeedbackResponse：定义这一接口对象的字段和类型约束。
# 未设默认值的字段需由调用方提供：id、repo_id、conversation_id、assistant_message_id、rating、review_status、notification_status、created_at、updated_at。
class ChatFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    # 明确资料所属仓库；跨仓库请求仍需服务端按归属校验。
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID
    run_id: UUID | None = None
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: str | None = None
    review_status: ReviewStatus
    review_note: str | None = None
    notification_status: str
    notification_error: str | None = None
    notified_at: datetime | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
