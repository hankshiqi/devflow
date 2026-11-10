# 学习导读｜导出 Skill 注册中心的公共类型与读取入口，供 API 和 Agent 共用。
from app.services.skills.registry import SkillActivation, SkillDefinition, SkillManifest, SkillRegistry, SkillRegistryError, get_skill_registry

__all__ = ["SkillActivation", "SkillDefinition", "SkillManifest", "SkillRegistry", "SkillRegistryError", "get_skill_registry"]
