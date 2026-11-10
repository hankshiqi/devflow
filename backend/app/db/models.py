# 学习导读｜关系数据库的 ORM 表结构：仓库、Issue、PR、CI、知识片段、会话、工作流和审计。
# Mapped 声明 Python 属性类型；mapped_column 定义列；ForeignKey 定义外键；relationship 定义对象关联。
# Repository 是主要的数据隔离入口；Document 保存文本与元数据，向量本身由 Milvus 保存。
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# 用户及演示角色信息，权限检查以实际角色规则为准。
# 映射到 users 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class User(Base):
    __tablename__ = "users"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), default="demo-user")
    role: Mapped[str] = mapped_column(String(80), default="owner", index=True)
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 已连接的仓库，是 Issue、PR、CI、文档和会话的数据归属边界。
# 映射到 repositories 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class Repository(Base):
    __tablename__ = "repositories"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner: Mapped[str] = mapped_column(String(200), index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    full_name: Mapped[str] = mapped_column(String(400), unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(80), default="github")
    api_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    clone_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkout_mode: Mapped[str] = mapped_column(String(80), default="managed")
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)
    github_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ORM 对象关联：通过 issues 访问关联对象，与对方的 repo 属性双向对应。
    # ORM 级联：父对象删除或子对象脱离父关联时清理子记录；这不同于数据库外键的 ON DELETE。
    issues: Mapped[list["Issue"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    # ORM 对象关联：通过 pull_requests 访问关联对象，与对方的 repo 属性双向对应。
    # ORM 级联：父对象删除或子对象脱离父关联时清理子记录；这不同于数据库外键的 ON DELETE。
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="repo", cascade="all, delete-orphan")
    # ORM 对象关联：通过 workflow_runs 访问关联对象，与对方的 repo 属性双向对应。
    # ORM 级联：父对象删除或子对象脱离父关联时清理子记录；这不同于数据库外键的 ON DELETE。
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(back_populates="repo", cascade="all, delete-orphan")


# 多个仓库的逻辑集合，供跨仓库汇总与报告使用。
# 映射到 workspaces 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class Workspace(Base):
    __tablename__ = "workspaces"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    repo_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    # 数据库外键：关联 users.id，实际保存的是对方记录的标识。
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 项目文档扫描指纹、状态、数量和稍后提醒时间。
# ProjectIndexState 会为每个仓库保存一条状态记录，用来表示索引当前进行到哪一步。
# 映射到 project_index_states 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ProjectIndexState(Base):
    # 数据库表名：模型与物理表通过它绑定，改表名只改这一行。
    __tablename__ = "project_index_states"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    # unique=True 让「一个仓库只能有一条状态记录」，因此这张表与 repositories 实际是一对一关系。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)
    # 状态机：missing(未索引) / building(构建中) / ready(已就绪) / failed(失败)。
    # 另有 stale(指纹已变) 由读取层临时派生，不一定写回数据库；索引 index 支持按状态筛仓库。
    status: Mapped[str] = mapped_column(String(40), default="missing", index=True)
    # 轻量变更指纹：对「文件路径 + 修改时间(纳秒) + 文件大小」做哈希，形如 files:<hex>。
    # 用途是「快而不绝对准」地判断要不要重扫；不读文件内容，所以内容变了但 mtime/size 未变时会漏判。
    fingerprint: Mapped[str] = mapped_column(String(200), default="")
    # 最近一次「成功」扫描的完成时间；失败分支不写这个字段，因此它代表最后一次好结果的时间。
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 暂缓索引的截止时间：在此刻之前调度器跳过该仓库，避免用户忽略后又反复扫描。
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # 本轮实际写入的 Document 条数（经过策略过滤、去重之后的真实落地数量）。
    docs_indexed: Mapped[int] = mapped_column(Integer, default=0)
    # 本轮扫描发现的候选文档总数（过滤之前的原始数量）。
    # 与 docs_indexed 的差值体现「发现了多少但被策略排除」，可直接展示为进度分母。
    docs_total: Mapped[int] = mapped_column(Integer, default=0)
    # 最近一次失败的异常文本，用于页面展示失败原因；成功后会被重置为 None。
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 项目概览摘要（技术栈、目录结构等），整体以 JSON 存放。
    # 结构由扫描逻辑决定、可能扩展，所以不拆成多个列，避免每加一个字段就改表结构。
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # 创建时间：INSERT 时由数据库 server_default 生成，应用层不赋值，避免多实例时钟不一致。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 更新时间：INSERT 时同样由数据库生成；UPDATE 时 onupdate 自动刷新，无需手写赋值。
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # ORM 对象关联：通过 repo 访问关联对象。
    # 这里是一对一（外键列唯一），SQLAlchemy 通过外键自动推导连接条件，无需写 primaryjoin。
    repo: Mapped[Repository] = relationship()


# 某个仓库下的一段聊天线程；消息与记忆通过会话 ID 归属。
# 映射到 conversations 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class Conversation(Base):
    __tablename__ = "conversations"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="默认会话")
    status: Mapped[str] = mapped_column(String(80), default="active", index=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)


# 同步后的 Issue 主信息，包括远程编号、正文、状态、标签和负责人。
# 映射到 issues 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class Issue(Base):
    __tablename__ = "issues"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    github_issue_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    number: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(50), default="open")
    labels: Mapped[list[str]] = mapped_column(JSON, default=list)
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    assignees: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ORM 对象关联：通过 repo 访问关联对象，与对方的 issues 属性双向对应。
    repo: Mapped[Repository] = relationship(back_populates="issues")


# PR 主信息和分支/时间状态；变更文件与行级评论使用关联子表。
# 映射到 pull_requests 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class PullRequest(Base):
    __tablename__ = "pull_requests"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    github_pr_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    number: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(50), default="open")
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    base_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)
    head_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ORM 对象关联：通过 repo 访问关联对象，与对方的 pull_requests 属性双向对应。
    repo: Mapped[Repository] = relationship(back_populates="pull_requests")
    # ORM 对象关联：通过 files 访问关联对象，与对方的 pr 属性双向对应。
    # ORM 级联：父对象删除或子对象脱离父关联时清理子记录；这不同于数据库外键的 ON DELETE。
    files: Mapped[list["PRFile"]] = relationship(back_populates="pr", cascade="all, delete-orphan")
    # ORM 对象关联：通过 review_comments 访问关联对象，与对方的 pr 属性双向对应。
    # ORM 级联：父对象删除或子对象脱离父关联时清理子记录；这不同于数据库外键的 ON DELETE。
    review_comments: Mapped[list["PRReviewComment"]] = relationship(back_populates="pr", cascade="all, delete-orphan")


# PR 中一个变更文件，patch 保存增删行差异，不是完整文件内容。
# 映射到 pr_files 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class PRFile(Base):
    __tablename__ = "pr_files"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 pull_requests.id，实际保存的是对方记录的标识。
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(50))
    additions: Mapped[int] = mapped_column(Integer, default=0)
    deletions: Mapped[int] = mapped_column(Integer, default=0)
    # 远程 PR 文件接口提供的增删行文本；允许为空，不能据此保证完整变更已获取。
    patch: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ORM 对象关联：通过 pr 访问关联对象，与对方的 files 属性双向对应。
    pr: Mapped[PullRequest] = relationship(back_populates="files")


# PR 某个文件/行上的 Review 评论，保留远程评论 ID 和定位信息。
# 映射到 pr_review_comments 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class PRReviewComment(Base):
    __tablename__ = "pr_review_comments"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 pull_requests.id，实际保存的是对方记录的标识。
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)
    github_comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ORM 对象关联：通过 pr 访问关联对象，与对方的 review_comments 属性双向对应。
    pr: Mapped[PullRequest] = relationship(back_populates="review_comments")


# 一次 CI 运行，包含状态、结论、Job/Step 和日志等信息。
# 映射到 workflow_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    github_run_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    name: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(80))
    conclusion: Mapped[str | None] = mapped_column(String(80), nullable=True)
    html_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    jobs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    logs_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ORM 对象关联：通过 repo 访问关联对象，与对方的 workflow_runs 属性双向对应。
    repo: Mapped[Repository] = relationship(back_populates="workflow_runs")


# Issue/PR/CI 等实体的一次分析结果，保存结构化输出便于回看。
# 映射到 analysis_results 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_type: Mapped[str] = mapped_column(String(80), index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)
    analysis_type: Mapped[str] = mapped_column(String(100), index=True)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 用于检索的文本片段；id 同步作为 Milvus 主键，meta 保存来源与向量生成信息。
# 映射到 documents 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class Document(Base):
    __tablename__ = "documents"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    source_type: Mapped[str] = mapped_column(String(80), index=True)
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 仓库知识库的切片、Embedding、检索、重排和生成配置。
# 映射到 knowledge_base_configs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class KnowledgeBaseConfig(Base):
    __tablename__ = "knowledge_base_configs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)
    embedding_provider: Mapped[str] = mapped_column(String(80), default="openai_compatible")
    embedding_model: Mapped[str] = mapped_column(String(240), default="qwen3.7-text-embedding")
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=1024)
    retrieval_method: Mapped[str] = mapped_column(String(40), default="hybrid")
    rerank_enabled: Mapped[bool] = mapped_column(default=True)
    rerank_provider: Mapped[str] = mapped_column(String(80), default="qwen_api")
    rerank_model: Mapped[str] = mapped_column(String(240), default="qwen3-rerank")
    top_k: Mapped[int] = mapped_column(Integer, default=3)
    score_threshold_enabled: Mapped[bool] = mapped_column(default=True)
    score_threshold: Mapped[float] = mapped_column(Float, default=0.8)
    chunk_size: Mapped[int] = mapped_column(Integer, default=800)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 上传的原始文件与处理状态，checksum 用于内容去重；一个源文件可生成多条 Document。
# 映射到 knowledge_source_documents 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class KnowledgeSourceDocument(Base):
    __tablename__ = "knowledge_source_documents"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    name: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(80), default="upload_file")
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    stored_path: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # 源文件内容指纹，用于识别重复/变更；不是文件路径，也不是片段主键。
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    character_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 一次召回测试的查询、配置和结果，便于检查检索效果。
# 映射到 retrieval_test_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class RetrievalTestRun(Base):
    __tablename__ = "retrieval_test_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    query: Mapped[str] = mapped_column(Text)
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 某版本源码中的函数/类等符号，含路径和行号定位。
# 映射到 code_symbols 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class CodeSymbol(Base):
    __tablename__ = "code_symbols"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    path: Mapped[str] = mapped_column(String(500), index=True)
    name: Mapped[str] = mapped_column(String(300), index=True)
    kind: Mapped[str] = mapped_column(String(80), default="symbol", index=True)
    language: Mapped[str] = mapped_column(String(80), default="text", index=True)
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    commit_sha: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 代码符号之间的关系，供影响分析和代码图展示。
# 映射到 code_relations 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class CodeRelation(Base):
    __tablename__ = "code_relations"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    # 数据库外键：关联 code_symbols.id，实际保存的是对方记录的标识。
    source_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)
    # 数据库外键：关联 code_symbols.id，实际保存的是对方记录的标识。
    target_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)
    source_name: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)
    target_name: Mapped[str] = mapped_column(String(300), index=True)
    relation_type: Mapped[str] = mapped_column(String(80), default="calls", index=True)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 原始用户/助手消息及工具、用量和上下文元数据。
# 映射到 chat_messages 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ChatMessage(Base):
    __tablename__ = "chat_messages"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(40), index=True)
    content: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 封存的一轮/段对话及其事件，用于后续精确读取与追踪。
# 映射到 chat_sessions 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    route: Mapped[str] = mapped_column(String(100), default="unknown")
    transcript_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    digest: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(80), default="sealed", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 线程级的结构化长期状态，保留任务、约束和决策。
# 映射到 thread_memories 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ThreadMemory(Base):
    __tablename__ = "thread_memories"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 会话记忆记录，与原始消息及即时模型上下文分别存储。
# 映射到 conversation_memories 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), unique=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 可搜索的仓库/会话证据投影，统一不同来源的读取格式。
# 映射到 evidence_items 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(80), index=True)
    source_id: Mapped[str] = mapped_column(String(120), index=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 业务知识图谱的节点关系，与向量近邻索引结构不同。
# 映射到 knowledge_graph_edges 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class KnowledgeGraphEdge(Base):
    __tablename__ = "knowledge_graph_edges"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    from_type: Mapped[str] = mapped_column(String(80), index=True)
    from_id: Mapped[str] = mapped_column(String(180), index=True)
    to_type: Mapped[str] = mapped_column(String(80), index=True)
    to_id: Mapped[str] = mapped_column(String(180), index=True)
    relation: Mapped[str] = mapped_column(String(80), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(120), default="explicit")
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 一次知识/记忆召回的查询与命中记录，用于解释检索行为。
# 映射到 recall_events 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class RecallEvent(Base):
    __tablename__ = "recall_events"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String(120), index=True)
    query: Mapped[str] = mapped_column(Text, default="")
    scope: Mapped[str] = mapped_column(String(120), default="project")
    mode: Mapped[str] = mapped_column(String(80), default="hybrid")
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


# 待审核的长期记忆候选，批准前不等于已确认知识。
# 映射到 memory_candidates 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class MemoryCandidate(Base):
    __tablename__ = "memory_candidates"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    # 数据库外键：关联 chat_sessions.id，实际保存的是对方记录的标识。
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(80), index=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(80), default="pending", index=True)
    source: Mapped[str] = mapped_column(String(120), default="session_sealer")
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 一次聊天 Agent 的问题、工具调用、最终回答和执行状态。
# 映射到 agent_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class AgentRun(Base):
    __tablename__ = "agent_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    user_message: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(100))
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    final_answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(80), default="success")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 对助手消息的反馈及审核信息，关联原消息/运行以便排错。
# 映射到 chat_feedback 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ChatFeedback(Base):
    __tablename__ = "chat_feedback"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"), index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    # 数据库外键：关联 chat_messages.id，实际保存的是对方记录的标识。
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_messages.id", ondelete="CASCADE"), unique=True, index=True
    )
    # 数据库外键：关联 agent_runs.id，实际保存的是对方记录的标识。
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True, index=True)
    # 数据库外键：关联 users.id，实际保存的是对方记录的标识。
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    rating: Mapped[str] = mapped_column(String(40), index=True)
    reason: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(40), default="open", index=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    notification_status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    notification_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True
    )


# 一次工程工作流计划与汇总结果，作为多个任务的父记录。
# 映射到 agent_workflow_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class AgentWorkflowRun(Base):
    __tablename__ = "agent_workflow_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    # 数据库外键：关联 conversations.id，实际保存的是对方记录的标识。
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)
    goal: Mapped[str] = mapped_column(Text)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    observation_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    final_answer: Mapped[str] = mapped_column(Text, default="")
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 工作流中单个 Agent 任务的输入、输出、依赖与执行状态。
# 映射到 agent_task_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class AgentTaskRun(Base):
    __tablename__ = "agent_task_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 agent_workflow_runs.id，实际保存的是对方记录的标识。
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_workflow_runs.id"), index=True)
    task_id: Mapped[str] = mapped_column(String(120), index=True)
    agent_name: Mapped[str] = mapped_column(String(120), index=True)
    task_type: Mapped[str] = mapped_column(String(120), index=True)
    claim_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# 一次评测的配置、进度、结果与诊断记录。
# 映射到 eval_runs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class EvalRun(Base):
    __tablename__ = "eval_runs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# 待用户确认的外部操作草稿；创建记录不会自动执行 GitHub 写入。
# 映射到 action_drafts 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class ActionDraft(Base):
    __tablename__ = "action_drafts"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)
    draft_type: Mapped[str] = mapped_column(String(100), index=True)
    target_type: Mapped[str] = mapped_column(String(100), index=True)
    target_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    risk_level: Mapped[str] = mapped_column(String(80), default="medium", index=True)
    status: Mapped[str] = mapped_column(String(80), default="pending_confirmation", index=True)
    # 数据库外键：关联 users.id，实际保存的是对方记录的标识。
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    # 数据库外键：关联 users.id，实际保存的是对方记录的标识。
    confirmed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 附加 JSON 元数据，如来源、路径、切片位置和 Embedding 生成信息；不是实际向量。
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# 权限敏感操作的审计信息，记录动作、目标和执行结果。
# 映射到 audit_logs 表；继承 Base 后，这些列和关系会被 SQLAlchemy 登记到 metadata。
class AuditLog(Base):
    __tablename__ = "audit_logs"

    # 本地主键：由 UUID 默认工厂生成，标识本表记录；与业务编号/远程 ID 的用途不同。
    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 数据库外键：关联 users.id，实际保存的是对方记录的标识。
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    # 数据库外键：关联 repositories.id，实际保存的是对方记录的标识。
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(160), index=True)
    target_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    target_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
