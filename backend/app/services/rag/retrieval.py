# 学习导读｜RAG 召回流水线：Milvus 向量召回、关系库文本 BM25、加权合并、去重、重排和压缩。
# hybrid 模式在 Milvus 不可用时保留关键词结果；纯 vector 模式会继续抛出不可用异常。
# 先取较多候选再筛选；分数是检索/排序信号，不是答案正确率。
import heapq
import hashlib
import math
import re
import uuid
from collections import Counter
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.context_compression import compress_retrieval_results
from app.services.rag.rerank import rerank_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES, is_searchable_source, normalize_source_types
from app.services.rag.vector_store import MilvusUnavailableError, search_milvus_documents


BM25_K1 = 1.5
BM25_B = 0.75
VECTOR_FUSION_WEIGHT = 0.62
KEYWORD_FUSION_WEIGHT = 0.30
DUAL_RECALL_BONUS = 0.08
MAX_CHUNKS_PER_PARENT = 2
NEAR_DUPLICATE_JACCARD = 0.9


# 计算稀疏词项向量的相似度，辅助关键词召回排序。
def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


# 将字符串/UUID 统一为数据库可比较的 UUID 类型。
def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


# 将数据库/接口中的时间值统一解析为 datetime；格式无法识别时返回空值。
def _parse_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = datetime.fromisoformat(value).date()
            return datetime.combine(parsed_date, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# 从候选 metadata 中取得可比较的时间戳，用于新鲜度排序。
def _metadata_time(doc_meta: dict) -> datetime | None:
    for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
        parsed = _parse_datetime(doc_meta.get(key))
        if parsed:
            return parsed
    return None


# 在 Python 中检查标签、状态、路径或时间等筛选条件；这是候选返回后的过滤。
def _matches_metadata(doc_meta: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    label = filters.get("label")
    if label and label not in (doc_meta.get("labels") or []):
        return False

    state = filters.get("state")
    if state and doc_meta.get("state") != state:
        return False

    path = (filters.get("path") or filters.get("path_prefix") or filters.get("module") or "").lower()
    if path:
        files = [str(item).lower() for item in doc_meta.get("files") or []]
        if files and not any(path in filename for filename in files):
            return False
        if not files and path not in " ".join(str(value).lower() for value in doc_meta.values()):
            return False

    start_at = _parse_datetime(filters.get("start_date") or filters.get("updated_after"))
    end_at = _parse_datetime(filters.get("end_date") or filters.get("updated_before"), end_of_day=True)
    doc_time = _metadata_time(doc_meta)
    if start_at and (doc_time is None or doc_time < start_at):
        return False
    if end_at and (doc_time is None or doc_time > end_at):
        return False

    for key, value in filters.items():
        if key in {
            "label",
            "path",
            "path_prefix",
            "module",
            "start_date",
            "end_date",
            "updated_after",
            "updated_before",
        } or value in (None, ""):
            continue
        if key == "branch":
            current = doc_meta.get("branch") or doc_meta.get("base_branch") or doc_meta.get("head_branch")
            if current != value:
                return False
            continue
        current = doc_meta.get(key)
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    return True


# 对输入文本分词/提取词项，为当前模块的匹配或排序提供信号。
def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


# 将用于摘要/提示的文本限制在长度预算内，原始业务记录由其他入口保存。
def _clip(text: str, limit: int = 260) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


# 把同一路候选得分缩放到可合并范围；归一化分数不是答案置信度。
def _normalise_scores(items: list[dict[str, Any]], field: str) -> None:
    if not items:
        return
    values = [max(0.0, float(item.get(field) or 0.0)) for item in items]
    max_value = max(values) or 1.0
    for item, value in zip(items, values):
        item[field] = round(value / max_value, 4)


# 将 SQL Document 转成统一检索结果字典，便于与 Milvus 候选使用同一格式。
def _document_result(doc: Document, score: float, source: str, score_field: str) -> dict[str, Any]:
    metadata = doc.meta or {}
    result = {
        "id": str(doc.id),
        "title": doc.title,
        "source_type": doc.source_type,
        "source_id": str(doc.source_id),
        "score": round(max(0.0, score), 4),
        "metadata": metadata,
        "snippet": _clip(doc.content),
        "_raw_content": doc.content,
        score_field: round(max(0.0, score), 4),
        "retrieval": {"sources": [source], score_field: round(max(0.0, score), 4)},
    }
    return result


# 从 Milvus 获取语义候选，并应用上层 metadata 条件和统一结果格式。
# 参数：repo_id：当前仓库标识，用于限定数据范围；query：用户问题或检索词。
# 参数：source_types：允许的来源类型集合；metadata_filters：对候选元数据施加的筛选条件。
def _milvus_vector_candidates(
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    #获取候选向量
    milvus_results = search_milvus_documents(
        repo_id,
        query,
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types if len(source_types) > 1 else None,
        limit=fetch_limit,
        query_embedding=query_embedding,
    )
    candidates = []
    for item in milvus_results:
        if not _matches_metadata(item.get("metadata") or {}, metadata_filters):
            continue
        score = max(0.0, float(item.get("score") or 0.0))
        candidates.append(
            {
                **item,
                "id": str(item.get("id") or item.get("source_id") or ""),
                "score": score,
                "vector_score": score,
                "vector_score_raw": score,
                "retrieval": {
                    "sources": ["vector"],
                    "vector_score_raw": score,
                    "vector_backend": "milvus",
                },
            }
        )
    _normalise_scores(candidates, "vector_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["vector_score"]
        item["retrieval"]["vector_score"] = item["vector_score"]
        item["vector_rank"] = rank
        item["retrieval"]["vector_rank"] = rank
    return candidates


# 统计文本长度和词项分布，为当前候选语料的 BM25 评分提供输入。
def _bm25_document_stats(doc: Document, query_terms: list[str]) -> dict[str, Any]:
    metadata = doc.meta or {}
    field_tokens = {
        "title": _tokens(doc.title or ""),
        "path": _tokens(str(metadata.get("path") or "")),
        "symbol": _tokens(str(metadata.get("symbol") or metadata.get("name") or "")),
        "labels": _tokens(" ".join(str(item) for item in metadata.get("labels") or [])),
        "content": _tokens(doc.content or ""),
    }
    field_weights = {"title": 4.0, "path": 3.0, "symbol": 4.5, "labels": 2.5, "content": 1.0}
    counters = {field: Counter(tokens) for field, tokens in field_tokens.items()}
    #计算文档中每个词的权重
    weighted_tf = {
        term: sum(float(counters[field].get(term, 0)) * field_weights[field] for field in field_tokens)
        for term in query_terms
    }
    #匹配的查询词
    matched_terms = [term for term in query_terms if weighted_tf.get(term, 0.0) > 0.0]
    #文档长度
    document_length = max(1, sum(len(tokens) for tokens in field_tokens.values()))
    return {
        "doc": doc,
        "weighted_tf": weighted_tf,
        "matched_terms": matched_terms,
        "document_length": document_length,
    }


# 结合词频、逆文档频率和长度归一化计算关键词相关性。
def _bm25_score(
    stats: dict[str, Any],
    *,
    document_count: int,
    document_frequency: Counter[str],
    average_document_length: float,
) -> float:
    score = 0.0
    document_length = float(stats["document_length"])
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / max(average_document_length, 1.0))
    for term, weighted_tf in stats["weighted_tf"].items():
        if weighted_tf <= 0:
            continue
        df = float(document_frequency.get(term, 0))
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))
        score += idf * (weighted_tf * (BM25_K1 + 1.0)) / (weighted_tf + length_norm)
    return max(0.0, score)


# 从 SQL 读取当前仓库允许来源的文档，在 Python 中进行 BM25 关键词评分。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：query：用户问题或检索词；source_types：允许的来源类型集合。
# 参数：metadata_filters：对候选元数据施加的筛选条件。
def _keyword_candidates(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
) -> list[dict[str, Any]]:
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return []
    q = db.query(Document).filter(
        Document.repo_id == _as_uuid(repo_id),
        Document.source_type.in_(source_types),
    )
    # 先把支持的状态条件压入 SQL 候选查询，减少不符合筛选条件的文档参与 BM25。
    state = (metadata_filters or {}).get("state")
    if state:
        q = q.filter(Document.meta["state"].as_string() == str(state))

    corpus: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    total_document_length = 0
    # 分批读取仓库文本记录，避免一次把整个文档集合都加载进 Python 内存。
    for doc in q.yield_per(500):
        if not _matches_metadata(doc.meta or {}, metadata_filters):
            continue
        #对文档进行加工stats产出如下注释
        stats = _bm25_document_stats(doc, terms)
        #
        #         {
        #   "doc": doc,                        # 原文档对象，后面组装结果要用
        #   "weighted_tf": {"timeout": 8.5, "超时": 4.0, ...},   # 每个查询词的加权词频
        #   "matched_terms": ["timeout", "超时"],                 # 实际命中了哪些词
        #   "document_length": 312,                              # 分词后的总长度
        # }
        corpus.append(stats)

        total_document_length += int(stats["document_length"])
        #统计每个查询词出现在多少篇文档里，这就是 BM25 的 df。{"timeout": 3, "超时": 12, "milvus": 47, "配置": 89}
        document_frequency.update(set(stats["matched_terms"]))

    if not corpus:
        return []

    average_document_length = total_document_length / len(corpus)
    top_candidates: list[tuple[float, int, dict[str, Any]]] = []
    # 利用当前仓库候选语料的词频统计计算每篇文档的 BM25 关键词相关度。
    for sequence, stats in enumerate(corpus):
        doc = stats["doc"]
        score = _bm25_score(
            stats,
            document_count=len(corpus),
            document_frequency=document_frequency,
            average_document_length=average_document_length,
        )
        if score <= 0:
            continue
        item = _document_result(doc, score, "keyword", "keyword_score")
        item["bm25_score"] = round(score, 4)
        item["bm25_matched_terms"] = list(stats["matched_terms"])
        item["retrieval"].update(
            {
                "keyword_backend": "bm25",
                "bm25_score": round(score, 4),
                "bm25_matched_terms": list(stats["matched_terms"]),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            }
        )
        entry = (score, sequence, item)
        if len(top_candidates) < fetch_limit:
            heapq.heappush(top_candidates, entry)
        elif score > top_candidates[0][0]:
            heapq.heapreplace(top_candidates, entry)
    candidates = [
        entry[2]
        for entry in sorted(top_candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ]
    _normalise_scores(candidates, "keyword_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["keyword_score"]
        item["retrieval"]["keyword_score"] = item["keyword_score"]
        item["keyword_rank"] = rank
        item["retrieval"]["keyword_rank"] = rank
    return candidates


# 按文档标识合并向量和关键词结果，使用权重及双路命中奖励重新排序。
def _merge_candidates(vector_items: list[dict[str, Any]], keyword_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    # 逐路合并向量和关键词候选，同时记录各路排名，供加权融合和检索追踪使用。
    for source_name, items in [("vector", vector_items), ("keyword", keyword_items)]:
        # 按父片段 ID 找同来源、同版本的邻居；范围校验用于避免把不同文档或旧版本拼在一起。
        for item in items:
            metadata = item.get("metadata") or {}
            key = str(
                metadata.get("chunk_id")
                or item.get("id")
                or f"{item.get('source_type')}:{item.get('source_id')}:{item.get('title')}"
            )
            current = merged.get(key)
            if current is None:
                current = {**item, "retrieval": {**(item.get("retrieval") or {})}}
                current["retrieval"]["sources"] = list(dict.fromkeys(current["retrieval"].get("sources") or [source_name]))
                merged[key] = current
            else:
                current["vector_score"] = max(float(current.get("vector_score") or 0.0), float(item.get("vector_score") or 0.0))
                current["keyword_score"] = max(float(current.get("keyword_score") or 0.0), float(item.get("keyword_score") or 0.0))
                current["vector_score_raw"] = max(
                    float(current.get("vector_score_raw") or 0.0), float(item.get("vector_score_raw") or 0.0)
                )
                current["bm25_score"] = max(float(current.get("bm25_score") or 0.0), float(item.get("bm25_score") or 0.0))
                current["snippet"] = max([str(current.get("snippet") or ""), str(item.get("snippet") or "")], key=len)
                current["_raw_content"] = max(
                    [str(current.get("_raw_content") or ""), str(item.get("_raw_content") or "")], key=len
                )
                current["retrieval"]["sources"] = list(dict.fromkeys([*(current["retrieval"].get("sources") or []), source_name]))
                for trace_key, trace_value in (item.get("retrieval") or {}).items():
                    if trace_key == "sources":
                        continue
                    current["retrieval"][trace_key] = trace_value
    output = []
    for item in merged.values():
        vector_score = float(item.get("vector_score") or 0.0)
        keyword_score = float(item.get("keyword_score") or 0.0)
        both_bonus = DUAL_RECALL_BONUS if vector_score and keyword_score else 0.0
        combined = min(1.0, VECTOR_FUSION_WEIGHT * vector_score + KEYWORD_FUSION_WEIGHT * keyword_score + both_bonus)
        item["combined_score"] = round(combined, 4)
        item["fusion_score"] = round(combined, 4)
        item["score"] = round(combined, 4)
        item["retrieval"].update(
            {
                "fusion_method": "weighted",
                "fusion_weights": {
                    "vector": VECTOR_FUSION_WEIGHT,
                    "keyword": KEYWORD_FUSION_WEIGHT,
                    "dual_recall_bonus": DUAL_RECALL_BONUS,
                },
                "dual_recall_bonus": both_bonus,
                "combined_score": item["combined_score"],
                "fusion_score": item["fusion_score"],
            }
        )
        output.append(item)
    output.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    for rank, item in enumerate(output, start=1):
        item["fusion_rank"] = rank
        item["retrieval"]["fusion_rank"] = rank
    return output


# 规范化候选文本，生成内容去重比较值；这不是数据库主键。
def _normalised_fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", str(text or "").lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest() if normalised else ""


# 从候选文本构造归一化词集合，供关键词匹配和混合排序使用。
def _candidate_term_set(item: dict[str, Any]) -> set[str]:
    text = str(item.get("_raw_content") or item.get("snippet") or "")
    return set(_tokens(text))


# 限制内容去重的来源范围，避免把不同版本或不同来源中相似文本错误合并。
def _same_duplicate_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    left_parent = str(left_meta.get("parent_id") or "")
    right_parent = str(right_meta.get("parent_id") or "")
    if left_parent and right_parent:
        return left_parent == right_parent
    return (
        str(left.get("source_type") or "") == str(right.get("source_type") or "")
        and str(left.get("source_id") or "") == str(right.get("source_id") or "")
    )


# 移除相同范围内的重复候选，减少同一内容占满最终结果。
def _deduplicate_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    term_sets: dict[str, set[str]] = {}
    # 按父片段 ID 找同来源、同版本的邻居；范围校验用于避免把不同文档或旧版本拼在一起。
    for item in items:
        item_id = str(item.get("id") or "")
        raw_text = str(item.get("_raw_content") or item.get("snippet") or "")
        fingerprint = _normalised_fingerprint(raw_text)
        duplicate_of: dict[str, Any] | None = fingerprints.get(fingerprint) if fingerprint else None
        if duplicate_of is not None and not _same_duplicate_scope(item, duplicate_of):
            duplicate_of = None
        if duplicate_of is None:
            current_terms = _candidate_term_set(item)
            for existing in kept:
                if not _same_duplicate_scope(item, existing):
                    continue
                existing_id = str(existing.get("id") or "")
                existing_terms = term_sets.get(existing_id) or _candidate_term_set(existing)
                if not current_terms or not existing_terms:
                    continue
                union = current_terms | existing_terms
                similarity = len(current_terms & existing_terms) / max(len(union), 1)
                if similarity >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = existing
                    break
            term_sets[item_id] = current_terms
        if duplicate_of is not None:
            duplicate_id = str(duplicate_of.get("id") or "")
            duplicate_trace = duplicate_of.setdefault("retrieval", {})
            duplicate_trace.setdefault("duplicates_removed", []).append(item_id)
            removed.append({"id": item_id, "duplicate_of": duplicate_id})
            continue
        kept.append(item)
        if fingerprint:
            fingerprints[fingerprint] = item
    return kept, removed


# 限制同一父文档/片段占用过多候选位置，提升来源覆盖范围。
def _limit_parent_occupancy(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    # 按父片段 ID 找同来源、同版本的邻居；范围校验用于避免把不同文档或旧版本拼在一起。
    for item in items:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and counts[parent_id] >= MAX_CHUNKS_PER_PARENT:
            removed.append({"id": str(item.get("id") or ""), "parent_id": parent_id})
            continue
        kept.append(item)
        if parent_id:
            counts[parent_id] += 1
    return kept, removed


# 检查相邻片段是否属于相同来源版本，避免父上下文扩展混入旧内容。
def _same_version(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("branch", "commit_sha", "version", "checksum", "knowledge_document_id"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


# 为命中的子片段补充同父级/同版本的相邻内容，在预算内恢复完整上下文。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
def expand_parent_context(
    db: Session,
    repo_id: str | uuid.UUID,
    items: list[dict[str, Any]],
    *,
    neighbor_window: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    expanded_items = 0
    expanded_chunks = 0
    # 按父片段 ID 找同来源、同版本的邻居；范围校验用于避免把不同文档或旧版本拼在一起。
    for item in items:
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or "")
        child_index = metadata.get("child_index")
        source_id = item.get("source_id")
        source_type = str(item.get("source_type") or "")
        retrieval = dict(item.get("retrieval") or {})
        if not parent_id or child_index is None or not source_id or not source_type:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "parent metadata unavailable"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        try:
            source_uuid = _as_uuid(source_id)
        except (TypeError, ValueError):
            retrieval["parent_expansion"] = {"expanded": False, "reason": "invalid source id"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        siblings = (
            db.query(Document)
            .filter(
                Document.repo_id == _as_uuid(repo_id),
                Document.source_type == source_type,
                Document.source_id == source_uuid,
            )
            .all()
        )
        selected: list[Document] = []
        current_index = int(child_index)
        for sibling in siblings:
            sibling_meta = sibling.meta or {}
            if str(sibling_meta.get("parent_id") or "") != parent_id:
                continue
            if not _same_version(metadata, sibling_meta):
                continue
            sibling_index = int(sibling_meta.get("child_index") or 0)
            if abs(sibling_index - current_index) <= max(0, neighbor_window):
                selected.append(sibling)
        # 按子片段在原文中的顺序重排邻居，拼接出来的上下文才与原文顺序一致。
        selected.sort(key=lambda row: int((row.meta or {}).get("child_index") or 0))
        if not selected:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "no matching siblings"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parts: list[str] = []
        seen_parts: set[str] = set()
        expanded_ids: list[str] = []
        # 去掉空文本和重复内容后再拼接，避免邻近片段重叠造成重复上下文。
        for sibling in selected:
            content = str(sibling.content or "").strip()
            fingerprint = _normalised_fingerprint(content)
            if not content or fingerprint in seen_parts:
                continue
            seen_parts.add(fingerprint)
            parts.append(content)
            expanded_ids.append(str(sibling.id))
        if not parts:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "empty sibling content"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parent_expanded = len(expanded_ids) > 1 or str(item.get("id") or "") not in expanded_ids
        retrieval["parent_expansion"] = {
            "expanded": parent_expanded,
            "parent_id": parent_id,
            "chunk_ids": expanded_ids,
            "neighbor_window": max(0, neighbor_window),
        }
        next_metadata = {**metadata, "expanded_chunk_ids": expanded_ids}
        expanded.append(
            {
                **item,
                "metadata": next_metadata,
                "_raw_content": "\n\n".join(parts),
                "parent_expanded": parent_expanded,
                "retrieval": retrieval,
            }
        )
        if parent_expanded:
            expanded_items += 1
            expanded_chunks += max(0, len(expanded_ids) - 1)
    return expanded, {
        "name": "parent_expansion",
        "input_count": len(items),
        "output_count": len(expanded),
        "expanded_items": expanded_items,
        "additional_chunks": expanded_chunks,
    }


# 按 vector/fulltext/hybrid 选择召回分支，再执行融合、去重、父级限额、重排及压缩。
# 参数：db：数据库会话，由调用方管理提交与关闭；repo_id：当前仓库标识，用于限定数据范围。
# 参数：query：用户问题或检索词；source_type：单一来源类型。
# 参数：limit：本次返回数量上限；metadata_filters：对候选元数据施加的筛选条件。
def search_similar_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 5,
    metadata_filters: dict | None = None,
    query_embedding: list[float] | None = None,
    retrieval_method: str = "hybrid",
    apply_rerank: bool = True,
    apply_compression: bool = True,
    source_types: list[str] | None = None,
    pipeline_trace: dict[str, Any] | None = None,
) -> list[dict]:
    # 先把调用方的来源类型统一规范化；后续过滤始终限定在允许检索的来源集合中。
    requested_types = normalize_source_types([source_type, *(source_types or [])]) or sorted(RAG_SOURCE_TYPES)
    searchable_types = list(dict.fromkeys(item for item in requested_types if is_searchable_source(item)))
    if not searchable_types:
        return []

    # 先超额召回，为 metadata 筛选、去重和重排预留候选；不是最终返回数量。
    fetch_limit = min(max(limit * 8, 40), 200)
    method = retrieval_method.strip().lower()
    vector_types = searchable_types
    vector_candidates: list[dict[str, Any]] = []
    keyword_candidates: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    vector_status = "skipped"
    if method in {"vector", "hybrid"} and vector_types:
        try:
            vector_candidates = _milvus_vector_candidates(
                repo_id,
                query,
                vector_types,
                fetch_limit,
                metadata_filters,
                query_embedding=query_embedding,
            )
            vector_status = "success"
        # hybrid 可继续使用关键词分支；vector 模式没有替代结果，需把异常交给调用方。
        except MilvusUnavailableError:
            vector_status = "unavailable"
            if method == "vector":
                raise
    stages.append(
        {
            "name": "vector_recall",
            "status": vector_status,
            "candidate_count": len(vector_candidates),
            "backend": "milvus",
        }
    )
    keyword_types = searchable_types if method in {"fulltext", "hybrid"} else []
    if keyword_types:
        # 关键词分支读取同仓库 SQL 文本并在 Python 中计算 BM25；这里不是 PostgreSQL 原生全文索引查询。
        keyword_candidates = _keyword_candidates(
            db,
            repo_id,
            query,
            keyword_types,
            fetch_limit,
            metadata_filters,
        )
    stages.append(
        {
            "name": "keyword_recall",
            "status": "success" if keyword_types else "skipped",
            "candidate_count": len(keyword_candidates),
            "backend": "bm25",
        }
    )
    if method == "vector":
        merged = vector_candidates
    elif method == "fulltext":
        merged = keyword_candidates
    else:
        # 双路结果按统一文档标识融合，保留向量与词法各自贡献，再进入去重/重排。
        merged = _merge_candidates(vector_candidates, keyword_candidates)
    stages.append(
        {
            "name": "weighted_fusion" if method == "hybrid" else "single_retriever_selection",
            "method": "weighted" if method == "hybrid" else method,
            "candidate_count": len(merged),
            "weights": {
                "vector": VECTOR_FUSION_WEIGHT,
                "keyword": KEYWORD_FUSION_WEIGHT,
                "dual_recall_bonus": DUAL_RECALL_BONUS,
            },
        }
    )
    # 融合后的候选先按文档身份去重，再限制同一父文档占比，避免重复片段挤满 Top-K。
    deduplicated, duplicates_removed = _deduplicate_candidates(merged)
    aggregated, parent_overflow_removed = _limit_parent_occupancy(deduplicated)
    stages.append(
        {
            "name": "deduplication_and_parent_aggregation",
            "input_count": len(merged),
            "output_count": len(aggregated),
            "duplicates_removed": duplicates_removed,
            "parent_overflow_removed": parent_overflow_removed,
        }
    )
    # 重排只作用于已经召回和过滤的候选；关闭重排时直接截取当前候选，无法补回漏召回文档。
    ranked = rerank_documents(query, aggregated, limit) if apply_rerank else aggregated[:limit]
    if apply_rerank:
        stages.append(
            {
                "name": "rerank",
                "input_count": len(aggregated),
                "output_count": len(ranked),
                "mode": ranked[0].get("retrieval", {}).get("rerank_mode") if ranked else None,
            }
        )
    if pipeline_trace is not None:
        pipeline_trace.update(
            {
                "retrieval_method": method,
                "fusion_method": "weighted" if method == "hybrid" else method,
                "stages": stages,
            }
        )
    if not apply_compression:
        return ranked
    # 最后压缩送入模型的片段长度；检索记录仍保留排序阶段信息，压缩不等于重新检索。
    compressed = compress_retrieval_results(query, ranked, max_tokens_per_item=260)
    if pipeline_trace is not None:
        pipeline_trace["stages"].append(
            {
                "name": "context_compression",
                "input_count": len(ranked),
                "output_count": len(compressed),
            }
        )
    return compressed
