# 学习导读｜HTTP 接口层：根据指定时间范围查询 Issue、PR 和 CI，生成周报与仓库健康摘要。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.reports import WeeklyReportRequest, WeeklyReportResponse
from app.services.agents.report_agent import ReportAgent
from app.services.rag.vector_store import add_document

router = APIRouter()


# 把日期范围转换为包含起止日的 UTC 时间边界。
def _range_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date, time.max, tzinfo=timezone.utc),
    )


# 统一时间值的时区格式，供比较前使用。
def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# 判断时间值是否处在报告统计窗口内。
def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = _normalize_dt(value)
    return normalized is not None and start_at <= normalized <= end_at


# 按创建、更新或关闭时间判断 Issue 是否属于报告区间。
def _issue_in_range(issue: Issue, start_at: datetime, end_at: datetime) -> bool:
    return any(_in_range(value, start_at, end_at) for value in [issue.created_at, issue.updated_at, issue.closed_at])


# 按创建、更新或合并时间判断 PR 是否属于报告区间。
def _pr_in_range(pr: PullRequest, start_at: datetime, end_at: datetime) -> bool:
    return any(_in_range(value, start_at, end_at) for value in [pr.created_at, pr.updated_at, pr.merged_at])


# 接口入口：POST /weekly；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 WeeklyReportResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.post("/weekly", response_model=WeeklyReportResponse)
async def weekly_report(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    start_at, end_at = _range_bounds(payload.start_date, payload.end_date)
    # 先按请求时间窗口筛选 Issue、PR 和 CI，再把这批本地数据交给报告 Agent 汇总。
    issues = [
        item
        for item in db.query(Issue).filter(Issue.repo_id == payload.repo_id).all()
        if _issue_in_range(item, start_at, end_at)
    ]
    prs = [
        item
        for item in db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id).all()
        if _pr_in_range(item, start_at, end_at)
    ]
    runs = [
        item
        for item in db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id).all()
        if _in_range(item.created_at, start_at, end_at) or _in_range(item.updated_at, start_at, end_at)
    ]
    merged_prs = [item for item in prs if item.merged_at]
    open_prs = [item for item in prs if item.state == "open"]
    failed_ci = [item for item in runs if item.conclusion == "failure"]
    # 报告 Agent 只基于上面准备好的结构化数据生成内容，不在这里同步 GitHub。
    result = await ReportAgent().run(
        {
            "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in issues],
            "pull_requests": [
                {"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at}
                for item in prs
            ],
            "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in runs],
            "metrics": {
                "issues": len(issues),
                "pull_requests": len(prs),
                "merged_prs": len(merged_prs),
                "open_prs": len(open_prs),
                "failed_ci": len(failed_ci),
            },
        },
        {
            "repo_id": str(payload.repo_id),
            "repo_name": repo.full_name,
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
        },
    )
    add_document(
        db,
        payload.repo_id,
        "weekly_report",
        None,
        f"研发周报 {payload.start_date} - {payload.end_date}",
        result["report_markdown"],
        {
            "display_title": f"研发周报 {payload.start_date} - {payload.end_date}",
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
            "repo_name": repo.full_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


# 接口入口：POST /repo-health；完整 URL 还需加上 main.py 注册的路由前缀。
@router.post("/repo-health")
async def repo_health(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    open_issues = db.query(Issue).filter(Issue.repo_id == payload.repo_id, Issue.state == "open").count()
    open_prs = db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id, PullRequest.state == "open").count()
    failed_ci = db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id, WorkflowRun.conclusion == "failure").count()
    return {
        "summary": "仓库健康度以 demo 指标估算，建议结合近期失败 CI 和高风险 PR 判断。",
        "metrics": {"open_issues": open_issues, "open_prs": open_prs, "failed_ci_runs": failed_ci, "merged_prs_7d": 0},
        "risks": ["存在失败 CI，需要排查"] if failed_ci else [],
    }
