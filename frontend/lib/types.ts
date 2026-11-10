// 学习导读｜跨页面共享的数据类型；TypeScript 类型用于开发检查，不能替代后端运行时校验。
// 已连接仓库的前端投影，字段对应后端仓库响应，不是 Git 客户端对象。
export type Repository = {
  id: string;
  owner: string;
  name: string;
  full_name: string;
  provider?: string;
  api_base_url?: string;
  clone_url?: string | null;
  local_path?: string | null;
  checkout_mode?: string;
  description?: string;
  default_branch?: string;
  last_sync_at?: string | null;
};
