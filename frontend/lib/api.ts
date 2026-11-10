// 学习导读｜统一 HTTP 客户端：JSON 请求、文件上传、错误提取和 SSE 流式事件解析。
// API_BASE_URL 来自构建时公开配置；这里只封装传输，不实现具体 Agent 业务。
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// 读取 JSON 数据并禁用请求缓存；泛型 T 只提供开发时类型检查。
export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 将请求体序列化为 JSON，提交后统一检查 HTTP 状态并解析响应。
export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 发送局部更新 JSON 请求，错误处理与其他请求保持一致。
export async function apiPatch<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 提交 FormData 上传文件；不手工设置 Content-Type，让浏览器自动生成 multipart boundary。
export async function apiUpload<T>(path: string, body: FormData): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body
  });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// POST 后持续读取响应流，缓存不完整事件，按 SSE 空行分隔后调用 onEvent。
export async function apiPostStream(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  options: { signal?: AbortSignal } = {}
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal
  });
  if (!response.ok) await throwApiError(response);
  if (!response.body) throw new Error("Streaming response body is not available.");

  // 读取的是网络字节块；一块可能包含多条 SSE，也可能只有一条事件的一部分。
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // 流式解码保留跨块的 UTF-8 字符，避免中文字符被拆开后乱码。
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    // 最后一段可能还没收到完整事件，留在 buffer 等待下一次 read，而不是立即 JSON.parse。
    buffer = events.pop() ?? "";
    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      const event = lines.find((line) => line.startsWith("event: "))?.slice(7) ?? "message";
      const dataText = lines
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      // 完整 data 解析后按事件名交给页面处理，例如 delta、tool_use、tool_result、final。
      if (dataText) onEvent(event, JSON.parse(dataText));
    }
  }
}

// 发送删除请求并解析后端结果，调用页面负责确认交互及刷新状态。
export async function apiDelete<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { method: "DELETE" });
  if (!response.ok) await throwApiError(response);
  return response.json();
}

// 优先提取后端 JSON 错误，再退回状态码/文本，统一抛出页面可显示的 Error。
async function throwApiError(response: Response): Promise<never> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    let message: string | null = null;
    try {
      const payload = await response.json();
      message = readApiErrorMessage(payload);
    } catch {
      message = null;
    }
    throw new Error(message || fallback);
  }
  const text = (await response.text()).trim();
  throw new Error(text || fallback);
}

// 兼容 FastAPI 字符串 detail、字段校验错误数组和普通 message 格式。
function readApiErrorMessage(payload: unknown): string | null {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return null;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) return String((item as { msg: unknown }).msg);
        return typeof item === "string" ? item : null;
      })
      .filter(Boolean);
    return messages.length ? messages.join("; ") : null;
  }
  const message = (payload as { message?: unknown }).message;
  return typeof message === "string" ? message : null;
}
