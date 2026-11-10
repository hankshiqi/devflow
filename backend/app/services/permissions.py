# 学习导读｜演示用户、角色权限检查与审计记录封装。
# 路由通过 require_permission 注入权限依赖，审计记录保存动作与目标。
from collections.abc import Callable

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import AuditLog, User
from app.db.session import get_db

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read", "settings:manage"},
    "maintainer": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read"},
    "developer": {"repo:read", "agent:run", "draft:create", "eval:read"},
    "viewer": {"repo:read", "eval:read"},
}


# 获取或创建本地演示用户，供审计记录和演示权限检查使用。
def ensure_demo_user(db: Session) -> User:
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is None:
        user = User(name="demo-owner", role="owner")
        db.add(user)
        db.commit()
        db.refresh(user)
    if not getattr(user, "role", None):
        user.role = "owner"
        db.commit()
        db.refresh(user)
    return user


# 判断当前用户角色是否包含指定权限。
def has_permission(user: User, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get((user.role or "viewer").lower(), set())


# 创建 FastAPI 权限依赖，供路由声明访问要求。
def require_permission(permission: str) -> Callable:
    # 从请求依赖取得用户和数据库，并在权限不足时拒绝请求。
    def dependency(db: Session = Depends(get_db)) -> User:
        user = ensure_demo_user(db)
        if not has_permission(user, permission):
            raise HTTPException(status_code=403, detail=f"需要权限：{permission}")
        return user

    return dependency


# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
def write_audit_log(
    db: Session,
    *,
    user: User | None,
    repo_id=None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    status: str = "success",
    request_json: dict | None = None,
    result_json: dict | None = None,
) -> AuditLog:
    row = AuditLog(
        user_id=user.id if user else None,
        repo_id=repo_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        status=status,
        request_json=request_json or {},
        result_json=result_json or {},
    )
    db.add(row)
    db.flush()
    return row
