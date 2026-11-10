// 学习导读｜RAG 页面共享类型和展示格式化函数，与后端知识库/检索响应字段对应。
// 检索方式枚举：向量、关键词 BM25 或双路混合。
export type RetrievalMethod = "vector" | "fulltext" | "hybrid";

// 知识库模型、切片、召回、重排和阈值配置。
export type KnowledgeConfig = {
  embedding_provider: string;
  embedding_model: string;
  embedding_dimensions: number;
  retrieval_method: RetrievalMethod;
  rerank_enabled: boolean;
  rerank_provider: string;
  rerank_model: string;
  top_k: number;
  score_threshold_enabled: boolean;
  score_threshold: number;
  chunk_size: number;
  chunk_overlap: number;
};

// 当前运行模型的状态描述，与可选模型目录分开。
export type RuntimeModel = {
  provider: string;
  model: string;
  available: boolean;
  fallback?: string | null;
  enabled?: boolean;
  dimensions?: number;
};

// 知识库概要及配置/运行信息，绑定真实仓库。
export type KnowledgeBase = {
  id: string;
  name: string;
  full_name: string;
  description: string;
  provider: string;
  document_count: number;
  chunk_count: number;
  character_count: number;
  config: KnowledgeConfig;
  runtime: { embedding: RuntimeModel; rerank: RuntimeModel };
  created_at?: string;
  updated_at?: string;
};

// 上传源文件及后台处理状态，一个源文件可能生成多个片段。
export type KnowledgeDocument = {
  id: string;
  repo_id: string;
  name: string;
  source_type: string;
  status: "queued" | "parsing" | "splitting" | "embedding" | "completed" | "failed";
  content_type?: string;
  file_size: number;
  character_count: number;
  chunk_count: number;
  error_message?: string;
  created_at?: string;
  completed_at?: string;
};

// 单个文档切片的正文和位置信息。
export type KnowledgeChunk = {
  id: string;
  position: number;
  title: string;
  content: string;
  character_count: number;
  enabled: boolean;
};

// 界面可选模型的能力描述，存在于目录不代表已下载到本机。
export type CatalogModel = {
  provider: string;
  model: string;
  label: string;
  available: boolean;
  recommended: boolean;
  fallback?: string | null;
  dimensions?: number;
};

// 可用 Embedding/重排模型目录，供配置表单选择。
export type ModelCatalog = {
  embedding_models: CatalogModel[];
  rerank_models: CatalogModel[];
  qwen: { embedding_available: boolean; rerank_available: boolean; note: string };
};

// 单条召回证据：来源、片段、分数与流水线 metadata。
export type RetrievalResult = {
  citation: number;
  document_id: string;
  source_id: string;
  source_type: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
  retrieval: Record<string, unknown>;
};

// 一次检索测试的候选、耗时和实际配置。
export type RetrievalResponse = {
  id: string;
  query: string;
  results: RetrievalResult[];
  result_count: number;
  took_ms: number;
  retrieval_config: KnowledgeConfig & {
    runtime?: unknown;
    fusion_method?: string;
    answer_gate?: {
      decision?: "answer" | "ask_clarification" | "insufficient_evidence" | "conflict";
      reason?: string;
      evidence_count?: number;
      top_score?: number;
    };
    pipeline?: {
      retrieval_method?: string;
      fusion_method?: string;
      stages?: Array<Record<string, unknown> & { name?: string }>;
    };
  };
  created_at?: string;
};

// 已保存召回测试的概要，供历史列表展示。
export type RetrievalHistory = {
  id: string;
  query: string;
  result_count: number;
  took_ms: number;
  created_at?: string;
};

export const statusLabel: Record<KnowledgeDocument["status"], string> = {
  queued: "排队中",
  parsing: "解析中",
  splitting: "分段中",
  embedding: "向量化",
  completed: "可用",
  failed: "失败"
};

export function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

// 将文件字节数转换为适合展示的容量单位。
export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

// 将时间值格式化供页面显示，缺失或无效时间按函数分支处理。
export function formatDate(value?: string) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
    new Date(value)
  );
}
