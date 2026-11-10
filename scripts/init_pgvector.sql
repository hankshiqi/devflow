-- 学习导读｜PostgreSQL 首次初始化时启用 pgvector 扩展。
-- 扩展可用不代表当前项目用它保存向量；现有 RAG 向量写入在 Milvus，SQL Document 保存文本与元数据。
CREATE EXTENSION IF NOT EXISTS vector;
