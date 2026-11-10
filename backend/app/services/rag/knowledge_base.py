# 学习导读｜管理知识库配置、模型契约和完整检索流程，并保存召回测试结果。
# 组合底层召回、可选远程重排、阈值、父片段扩展及上下文预算。
import importlib.util
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import KnowledgeBaseConfig, RetrievalTestRun
from app.services.context_compression import compress_retrieval_results, estimate_tokens
from app.services.rag.answer_gate import evaluate_answer_gate
from app.services.rag.embeddings import (
    embed_knowledge_query,
    embedding_contract,
    normalize_embedding_provider,
    validate_embedding_contract,
)
from app.services.rag.rerank import rerank_documents
from app.services.rag.retrieval import expand_parent_context, search_similar_documents
from app.services.rag.source_policy import normalize_source_types, should_vectorize


# 将接口中的字符串标识转成数据库 UUID，保证关联查询类型一致。
def as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# 读取仓库知识库配置；不存在时创建默认配置，后续检索和写入共用。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：commit：是否在本函数内提交 SQL 事务。
def get_or_create_config(db: Session, repo_id: str | uuid.UUID, *, commit: bool = True) -> KnowledgeBaseConfig:
    repo_uuid = as_uuid(repo_id)
    config = db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo_uuid).one_or_none()
    if config is None:
        contract = embedding_contract()
        config = KnowledgeBaseConfig(
            repo_id=repo_uuid,
            embedding_provider=contract["provider"],
            embedding_model=contract["model"],
            embedding_dimensions=contract["dimensions"],
        )
        db.add(config)
        db.flush()
        if commit:
            db.commit()
            db.refresh(config)
    validate_embedding_contract(
        config.embedding_provider,
        config.embedding_model,
        config.embedding_dimensions,
    )
    return config


# 把知识库 ORM 配置转换为 API/运行时使用的普通字典。
def config_dict(config: KnowledgeBaseConfig) -> dict[str, Any]:
    return {
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dimensions": config.embedding_dimensions,
        "retrieval_method": config.retrieval_method,
        "rerank_enabled": config.rerank_enabled,
        "rerank_provider": config.rerank_provider,
        "rerank_model": config.rerank_model,
        "top_k": config.top_k,
        "score_threshold_enabled": config.score_threshold_enabled,
        "score_threshold": config.score_threshold,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }


# 返回界面可选择的模型目录及能力信息，不会自动下载所有模型。
def model_catalog() -> dict[str, Any]:
    sentence_transformers_ready = importlib.util.find_spec("sentence_transformers") is not None
    qwen_api_ready = bool((settings.rerank_api_key or settings.dashscope_api_key) and settings.rerank_base_url)
    contract = embedding_contract()
    provider = normalize_embedding_provider(str(contract["provider"]))
    if provider in {"qwen_local", "local"}:
        embedding_available = sentence_transformers_ready
    elif provider == "dashscope":
        embedding_available = bool(settings.dashscope_api_key)
    elif provider == "openai_compatible":
        embedding_available = bool(settings.embedding_api_key or settings.llm_api_key)
    else:
        embedding_available = provider == "deterministic"
    return {
        "embedding_models": [
            {
                "provider": provider,
                "model": contract["model"],
                "label": f"{contract['model']}（固定向量空间）",
                "dimensions": contract["dimensions"],
                "available": embedding_available,
                "recommended": True,
                "fallback": None,
            },
        ],
        "rerank_models": [
            {
                "provider": "qwen_api",
                "model": settings.qwen_rerank_model,
                "label": "Qwen3-Rerank（百炼 API）",
                "available": qwen_api_ready,
                "recommended": True,
                "fallback": "heuristic",
            },
            {
                "provider": "heuristic",
                "model": "heuristic-reranker",
                "label": "本地启发式重排",
                "available": True,
                "recommended": False,
                "fallback": None,
            },
        ],
        "qwen": {
            "embedding_available": embedding_available,
            "rerank_available": qwen_api_ready,
            "note": "当前使用通义千问兼容 API；重排接口不可用时自动回退到本地启发式重排。",
        },
    }


# 检查知识库配置与依赖服务状态，供页面显示索引和检索是否可用。
def runtime_status(config: KnowledgeBaseConfig) -> dict[str, Any]:
    catalog = model_catalog()
    embedding = next(
        (
            item
            for item in catalog["embedding_models"]
            if item["provider"] == config.embedding_provider and item["model"] == config.embedding_model
        ),
        None,
    )
    rerank = next(
        (
            item
            for item in catalog["rerank_models"]
            if item["provider"] == config.rerank_provider and item["model"] == config.rerank_model
        ),
        None,
    )
    return {
        "embedding": {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "dimensions": config.embedding_dimensions,
            "available": bool(embedding and embedding["available"]),
            "fallback": (embedding or {}).get("fallback"),
        },
        "rerank": {
            "enabled": config.rerank_enabled,
            "provider": config.rerank_provider,
            "model": config.rerank_model,
            "available": bool(rerank and rerank["available"]),
            "fallback": (rerank or {}).get("fallback"),
        },
    }


# 调用配置的 Qwen 兼容重排服务，为已召回候选取得更细的相关性分数。
# 参数：query：用户问题或检索词；limit：本次返回数量上限。
async def _qwen_rerank(
    query: str,
    items: list[dict[str, Any]],
    config: KnowledgeBaseConfig,
    limit: int,
) -> list[dict[str, Any]]:
    api_key = settings.rerank_api_key or settings.dashscope_api_key
    base_url = (settings.rerank_base_url or "").rstrip("/")
    # 远程重排缺少密钥或地址时明确失败，让调用方按策略回退到本地启发式重排。
    if not api_key or not base_url:
        raise RuntimeError("Qwen rerank API is not configured")
    endpoint = base_url if base_url.endswith("/reranks") else f"{base_url}/reranks"
    documents = [
        "\n".join(
            part
            for part in (
                str(item.get("title") or ""),
                str((item.get("metadata") or {}).get("path") or ""),
                str(item.get("_raw_content") or item.get("snippet") or ""),
            )
            if part
        )[:4000]
        for item in items
    ]
    async with httpx.AsyncClient(timeout=60) as client:
        # 通过配置的兼容 API 发送查询和候选文本，HTTP 错误会交由外层 fallback 捕获。
        response = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": config.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(max(1, limit), len(documents)),
                "instruct": settings.qwen_retrieval_instruction,
            },
        )
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("results") or payload.get("output", {}).get("results") or []
    ranked: list[dict[str, Any]] = []
    # 逐条校验服务端返回的候选索引，跳过越界项并映射回原始文档对象。
    for row in rows:
        index = int(row.get("index", -1))
        if index < 0 or index >= len(items):
            continue
        score = float(row.get("relevance_score") or row.get("score") or 0.0)
        item = items[index]
        ranked.append(
            {
                **item,
                "score": round(score, 4),
                "rerank_score": round(score, 4),
                "rank_reason": "qwen3-rerank",
                "retrieval": {
                    **(item.get("retrieval") or {}),
                    "rerank_mode": "qwen_api",
                    "rerank_model": config.rerank_model,
                    "rerank_score": round(score, 4),
                },
            }
        )
    return ranked


# 知识库检索总入口：应用配置、召回与重排、阈值和父上下文扩展，并返回流水线诊断。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：query：用户问题或检索词；source_types：允许的来源类型集合。
# 参数：metadata_filters：对候选元数据施加的筛选条件。
async def retrieve_knowledge(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    *,
    top_k: int | None = None,
    retrieval_method: str | None = None,
    rerank_enabled: bool | None = None,
    score_threshold_enabled: bool | None = None,
    score_threshold: float | None = None,
    source_types: list[str] | None = None,
    metadata_filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    started = time.perf_counter()
    # 先取得仓库级知识库设置，后面的检索模式、Embedding 契约和重排策略都以此为准。
    config = get_or_create_config(db, repo_id)
    resolved_top_k = min(max(top_k or config.top_k, 1), 20)
    method = retrieval_method or config.retrieval_method
    #是否进行重排序
    use_rerank = config.rerank_enabled if rerank_enabled is None else rerank_enabled
    #是否启用分数筛选
    threshold_on = config.score_threshold_enabled if score_threshold_enabled is None else score_threshold_enabled
    #具体的分数阈值
    threshold = config.score_threshold if score_threshold is None else score_threshold


    #查询向量必须遵守入库时的 Embedding Contract。
    #它把调用者传入的来源类型整理干净：
    requested_source_types = normalize_source_types(source_types)
    #有效的来源
    effective_source_types = requested_source_types or None
    #判断当前筛选范围内是否存在允许生成向量的来源，只要有一个可以向量化就会进行向量化
    has_vector_sources = not effective_source_types or any(
        should_vectorize(source_type) for source_type in effective_source_types
    )
    #查询向量初始化为空
    query_embedding = None
    if method in {"vector", "hybrid"} and has_vector_sources:
        #向量或混合模式且来源允许向量化时才生成 query embedding；纯关键词路径不发起向量模型请求。
        #生成问题的嵌入向量
        query_embedding = embed_knowledge_query(
            query,
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )
    # 初次召回数量大于最终 Top-K，为后续重排、扩展和门槛筛选预留候选。
    candidate_limit = min(max(resolved_top_k * (8 if use_rerank else 2), 20), 100)
    pipeline_trace: dict[str, Any] = {}
    # 先执行底层向量/关键词/混合召回，并关闭底层重排和压缩，让本函数统一控制完整流水线。 返回召回得到的文档列表
    items = search_similar_documents(
        db,
        repo_id,
        query,
        limit=candidate_limit,
        query_embedding=query_embedding,
        retrieval_method=method,
        apply_rerank=False,
        apply_compression=False,
        source_types=effective_source_types,
        metadata_filters=metadata_filters,
        pipeline_trace=pipeline_trace,
    )
    rerank_stage: dict[str, Any] = {
        "name": "rerank",
        "input_count": len(items),
        "requested": "qwen_api" if use_rerank and config.rerank_provider == "qwen_api" else "heuristic",
    }
    #rerank
    if use_rerank and items:
        # 按配置选择 Qwen 远程重排；失败时进入启发式重排兜底，并在 trace 中标明原因。 rerank
        if config.rerank_provider == "qwen_api":
            try:
                items = await _qwen_rerank(query, items, config, resolved_top_k)
                if not items:
                    raise RuntimeError("Qwen rerank returned no valid results")
                rerank_stage.update({"mode": "qwen_api", "fallback": False})
            except Exception as exc:
                items = rerank_documents(query, items, resolved_top_k)
                rerank_stage.update(
                    {
                        "mode": "heuristic",
                        "fallback": True,
                        "fallback_reason": type(exc).__name__,
                    }
                )
        else:
            items = rerank_documents(query, items, resolved_top_k)
            rerank_stage.update({"mode": "heuristic", "fallback": False})
    elif use_rerank:
        rerank_stage.update({"mode": "not_run", "fallback": False, "reason": "no candidates"})
    else:
        rerank_stage.update({"mode": "disabled", "fallback": False})
    items = items[:resolved_top_k]
    pipeline_trace.setdefault("stages", []).append({**rerank_stage, "output_count": len(items)})
    #父级扩展
    # 对命中的子片段补充同源、同版本的邻近片段，帮助模型恢复局部上下文。 父级扩展
    items, parent_stage = expand_parent_context(db, repo_id, items, neighbor_window=1)
    pipeline_trace["stages"].append(parent_stage)

    before_threshold = len(items)
    #score筛选
    # 分数门槛在重排和父片段扩展后执行，并把筛选前后数量记入流水线追踪。
    if threshold_on:
        items = [item for item in items if float(item.get("score") or 0.0) >= threshold]
    pipeline_trace["stages"].append(
        {
            "name": "score_threshold",
            "enabled": threshold_on,
            "threshold": threshold if threshold_on else None,
            "input_count": before_threshold,
            "output_count": len(items),
        }
    )

    context_budget_tokens = min(1800, max(520, resolved_top_k * 260))
    compression_input_count = len(items)
    # 在送给生成模型前按总 Token 预算压缩片段，减少无关文本占用上下文。
    #压缩片段
    items = compress_retrieval_results(
        query,
        items,
        max_tokens_per_item=260,
        max_total_tokens=context_budget_tokens,
    )
    compressed_tokens = sum(estimate_tokens(str(item.get("snippet") or "")) for item in items)
    pipeline_trace["stages"].append(
        {
            "name": "context_compression",
            "input_count": compression_input_count,
            "output_count": len(items),
            "budget_tokens": context_budget_tokens,
            "selected_tokens": compressed_tokens,
        }
    )
    #answergate
    # 最终检查证据能否支撑回答；门控结论写入每条结果和 trace，供问答层决定回答或澄清。
    gate = evaluate_answer_gate(
        query,
        items,
        score_threshold=threshold,
        threshold_enabled=threshold_on,
    )
    for item in items:
        retrieval = dict(item.get("retrieval") or {})
        retrieval["answer_gate"] = gate["decision"]
        retrieval["selected_for_context"] = gate["decision"] == "answer"
        item["retrieval"] = retrieval
    pipeline_trace["stages"].append({"name": "answer_gate", **gate})
    duration_ms = round((time.perf_counter() - started) * 1000)
    trace = {
        **config_dict(config),
        "top_k": resolved_top_k,
        "source_types": requested_source_types,
        "filters": {
            "repo_id": str(repo_id),
            "source_types": requested_source_types,
            "metadata": metadata_filters or {},
        },
        "runtime": runtime_status(config),
        "fusion_method": "weighted" if method == "hybrid" else method,
        "context_budget_tokens": context_budget_tokens,
        "answer_gate": gate,
        "pipeline": pipeline_trace,
    }
    return items, trace, duration_ms


# 保存一次召回测试的查询、配置与结果，便于前端回看和比较。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：query：用户问题或检索词。
def record_retrieval_test(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    retrieval_config: dict[str, Any],
    results: list[dict[str, Any]],
    duration_ms: int,
) -> RetrievalTestRun:
    run = RetrievalTestRun(
        repo_id=as_uuid(repo_id),
        query=query,
        retrieval_config=retrieval_config,
        results_json=results,
        duration_ms=duration_ms,
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
