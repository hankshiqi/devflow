# 学习导读｜建立 SQLAlchemy 引擎、会话工厂与 ORM 声明基类。
# 请求通过 get_db 获取独立 Session；flush 发送 SQL，commit 才提交事务，close 释放连接资源。
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


# 引擎管理连接池；pool_pre_ping 在取连接时检查存活，减少数据库重启后的陈旧连接问题。
engine = create_engine(settings.database_url, pool_pre_ping=True)
# 关闭自动提交/自动 flush，由业务代码显式决定何时发送 SQL 和提交。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# FastAPI yield 依赖：创建请求级 Session，路由结束后无论成功失败都关闭会话。
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        # yield 前创建资源，yield 后由 FastAPI 在请求结束时执行 finally 清理；这里不自动 commit。
        yield db
    finally:
        db.close()
