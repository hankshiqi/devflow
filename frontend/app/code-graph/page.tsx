// 学习导读｜代码图查询界面：按仓库/PR 快照搜索符号与影响关系，展示分支和提交信息。
// 这里展示业务代码图，与 Milvus 的向量近邻图不同。
"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";

// PR 主信息及可选文件列表；状态来自后端同步结果。
type PullRequest = {
  id: string;
  number: number;
  title: string;
  head_branch?: string | null;
};

type CodeGraph = {
  query: string;
  symbols: Array<{ id: string; path: string; name: string; kind: string; start_line?: number | null; pr_number?: number | null }>;
  relations: Array<{ id: string; source_name?: string | null; target_name: string; relation_type: string; path?: string | null; pr_number?: number | null }>;
  documents: Array<{ id: string; title: string; path?: string; snippet: string; pr_number?: number | null }>;
};

type PrSnapshot = {
  status: string;
  message?: string | null;
  snapshot_name?: string | null;
  branch?: string | null;
  commit_sha?: string | null;
  pr_id?: string | null;
  pr_number?: number | null;
  indexed_symbols?: number | null;
};

// 代码图查询界面：按仓库/PR 快照搜索符号与影响关系，展示分支和提交信息。
export default function CodeGraphPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [prs, setPrs] = useState<PullRequest[]>([]);
  const [prId, setPrId] = useState("");
  const [query, setQuery] = useState("");
  const [graph, setGraph] = useState<CodeGraph | null>(null);
  const [snapshot, setSnapshot] = useState<PrSnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  async function loadPrs(id: string) {
    if (!id) return;
    const items = await apiGet<PullRequest[]>(`/api/repos/${id}/pull-requests`).catch(() => []);
    setPrs(items);
    setPrId(items[0]?.id || "");
    setSnapshot(null);
  }

  // 组件挂载后执行初始化；如返回清理函数，卸载时会释放监听器/定时器等资源。
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
        return loadPrs(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  // 提交符号或路径查询，展示后端代码图中的匹配节点和影响关系。
  async function searchGraph(snapshotPrId?: string) {
    if (!repoId) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ q: query });
      const activeSnapshotPrId = snapshotPrId ?? (snapshot?.status === "ready" ? prId : "");
      if (activeSnapshotPrId) params.set("pr_id", activeSnapshotPrId);
      setGraph(await apiGet<CodeGraph>(`/api/repos/${repoId}/code-graph/search?${params.toString()}`));
    } finally {
      setLoading(false);
    }
  }

  // 请求为指定 PR 准备代码快照，完成后可切换到对应版本分析。
  async function prepareSnapshot() {
    if (!prId) return;
    setLoading(true);
    try {
      const nextSnapshot = await apiPost<PrSnapshot>(`/api/pull-requests/${prId}/snapshot`, {});
      setSnapshot(nextSnapshot);
      await searchGraph(nextSnapshot.status === "ready" ? prId : undefined);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="代码图谱" description="检索代码 symbol、调用/导入关系，并为 PR 准备独立分支快照。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <label className="text-sm font-medium text-slate-700">
            仓库
            <select
              className="mt-2 w-full rounded-md border border-line px-3 py-2"
              value={repoId}
              onChange={(event) => {
                setRepoId(event.target.value);
                setGraph(null);
                loadPrs(event.target.value);
              }}
            >
              {repos.map((repo) => (
                <option key={repo.id} value={repo.id}>
                  {repo.full_name}
                </option>
              ))}
            </select>
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            搜索 symbol / path
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <PrimaryButton className="mt-3" disabled={!repoId || loading} onClick={() => searchGraph()}>
            搜索图谱
          </PrimaryButton>
          <div className="mt-6 border-t border-line pt-4">
            <label className="text-sm font-medium text-slate-700">
              PR 分支快照
              <select
                className="mt-2 w-full rounded-md border border-line px-3 py-2"
                value={prId}
                onChange={(event) => {
                  setPrId(event.target.value);
                  setSnapshot(null);
                  setGraph(null);
                }}
              >
                {prs.map((pr) => (
                  <option key={pr.id} value={pr.id}>
                    #{pr.number} {pr.title}
                  </option>
                ))}
              </select>
            </label>
            <PrimaryButton className="mt-3" disabled={!prId || loading} onClick={prepareSnapshot}>
              准备 PR 快照
            </PrimaryButton>
            {snapshot && (
              <div className="mt-3 rounded-md border border-line bg-panel p-3 text-sm text-slate-700">
                <div className="font-medium text-ink">{snapshot.snapshot_name || "PR 分支快照"}</div>
                <div className="mt-1">状态：{snapshot.status}</div>
                {snapshot.branch && <div>分支：{snapshot.branch}</div>}
                {snapshot.commit_sha && <div>Commit：{snapshot.commit_sha.slice(0, 12)}</div>}
                {typeof snapshot.indexed_symbols === "number" && <div>代码图符号：{snapshot.indexed_symbols}</div>}
                {snapshot.message && <div className="mt-1 text-slate-500">{snapshot.message}</div>}
              </div>
            )}
          </div>
        </Card>

        <div className="grid gap-4">
          <Card>
            <h2 className="font-semibold text-ink">Symbols</h2>
            <div className="mt-3 grid gap-2">
              {(graph?.symbols || []).map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm">
                  <span className="font-medium text-ink">{item.name}</span>
                  <span className="ml-2 text-slate-500">
                    {item.kind} · {item.path}:{item.start_line || 1}
                    {item.pr_number ? ` · PR #${item.pr_number}` : ""}
                  </span>
                </div>
              ))}
              {!graph && <div className="text-sm text-slate-500">索引代码后可搜索 symbol。</div>}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">Relations</h2>
            <div className="mt-3 grid gap-2">
              {(graph?.relations || []).slice(0, 30).map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm text-slate-700">
                  {item.source_name || item.path || "module"} - {item.relation_type} - {item.target_name}
                  {item.pr_number ? <span className="ml-2 text-slate-500">PR #{item.pr_number}</span> : null}
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">相关代码片段</h2>
            {(graph?.documents || []).map((item) => (
              <pre key={item.id} className="mt-3 max-h-48 overflow-auto rounded-md border border-slate-800 bg-slate-950 p-3 text-xs text-slate-100">
                {item.title}
                {"\n\n"}
                {item.snippet}
              </pre>
            ))}
          </Card>
        </div>
      </div>
    </>
  );
}
