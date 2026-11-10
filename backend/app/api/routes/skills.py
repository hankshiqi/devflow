# 学习导读｜HTTP 接口层：展示已发现的 Skill 定义与记忆 MCP 服务状态。
# FastAPI 从请求解析参数和 Pydantic 数据，并通过 Depends 注入数据库/权限依赖。
# 路由负责输入输出与上下文组织，底层业务实现尽量复用 services。
from fastapi import APIRouter, HTTPException

from app.schemas.skills import McpRuntimeResponse, McpToolResponse, SkillListResponse, SkillManifestResponse, SkillResponse
from app.services.mcp_client import McpClientError, StdioMcpClient
from app.services.skills import get_skill_registry

router = APIRouter()


# 将技能定义转换为页面响应；是否返回完整指令由 include_instructions 控制。
def _skill_response(skill, include_instructions: bool = False) -> SkillResponse:
    return SkillResponse(
        manifest=SkillManifestResponse.model_validate(skill.manifest.model_dump()),
        path=skill.path,
        instructions=skill.instructions if include_instructions else None,
    )


# 接口入口：GET ；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 SkillListResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    registry = get_skill_registry()
    return SkillListResponse(skills=[_skill_response(skill) for skill in registry.list_skills()])


# 接口入口：GET /mcp-status；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 McpRuntimeResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/mcp-status", response_model=McpRuntimeResponse)
async def get_mcp_status() -> McpRuntimeResponse:
    try:
        inspection = await StdioMcpClient().inspect_server()
    except (McpClientError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"MCP Server 验证失败：{exc}") from exc

    initialize = inspection.get("initialize") or {}
    server_info = initialize.get("serverInfo") or {}
    tools = [
        McpToolResponse(
            name=str(item.get("name") or ""),
            description=str(item.get("description") or ""),
            input_schema=item.get("inputSchema") or {},
        )
        for item in inspection.get("tools") or []
    ]
    return McpRuntimeResponse(
        status="online",
        server_name=str(server_info.get("name") or "unknown"),
        server_version=str(server_info.get("version") or "unknown"),
        protocol_version=str(initialize.get("protocolVersion") or "unknown"),
        transport="stdio",
        tools=tools,
    )


# 接口入口：GET /{skill_name}；完整 URL 还需加上 main.py 注册的路由前缀。
# 返回值按 SkillResponse 校验和序列化，避免把 ORM 内部字段直接暴露给页面。
@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str) -> SkillResponse:
    registry = get_skill_registry()
    skill = registry.get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail="未找到 Skill")
    return _skill_response(skill, include_instructions=True)
