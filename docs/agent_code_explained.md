# Agent 代码注释与执行流程说明

本文对应你粘贴的 Python 脚本。原脚本没有直接放在当前项目目录中，因此这里保留原代码不动，使用关键代码的“注释版片段”解释其职责、数据流和调用关系。

## 1. 这段程序整体做什么

它模拟了一个用于判断 GitHub Pull Request 是否可以合并的 Agent 系统。程序不会直接修改仓库，而是：

1. 从固定的演示数据中读取 PR、Issue、CI、Review 和发布规则；
2. 让大模型决定下一步应该读取哪类证据；
3. 把模型选择的动作限制在 `TOOLS` 白名单内；
4. 把工具结果写入 `EvidenceState`；
5. 根据失败 CI、未解决 Review 和高风险目录等确定性规则计算阻塞项；
6. 让模型汇总结论，并可选地再做一次 Reflection 审查。

因此它是一个“Agent 编排演示”，不是直接连接 GitHub 的生产实现。`PR_DATA`、`ISSUE_DATA`、`CI_DATA` 等目前都是内存中的模拟数据。

## 2. 数据模型：配置、轨迹和证据状态

```python
@dataclass
class ModelConfig:
    # OpenAI 兼容接口的地址，例如 https://.../v1
    base_url: str
    # 要调用的模型名称，例如 qwen-plus 或其他兼容模型
    model: str
    # API 密钥。这里只在请求头中使用，不应打印具体值
    api_key: str
    # 实际加载到的 .env 文件路径，便于诊断配置来源
    env_file: str | None = None


@dataclass
class TraceEvent:
    # 当前是第几步，按写入 trace 的顺序递增
    step: int
    # 谁发起了动作，例如 ReActAgent、PlannerAgent
    actor: str
    # 做了什么，例如 read_pr、synthesize
    action: str
    # 传给动作的标准化参数
    arguments: dict[str, Any] = field(default_factory=dict)
    # 动作执行后的结果
    observation: dict[str, Any] = field(default_factory=dict)
    # 对动作目的的简短说明
    thought: str = ""
```

`EvidenceState` 有两类数据，必须区分：

```python
@dataclass
class EvidenceState:
    # 当前已经拿到的最新证据
    pr: dict[str, Any] | None = None
    issue: dict[str, Any] | None = None
    ci: dict[str, Any] | None = None
    review_comments: list[dict[str, Any]] | None = None
    release_policy: dict[str, Any] | None = None

    # 历史轨迹。它记录“做过什么”，不是当前证据本身
    trace: list[TraceEvent] = field(default_factory=list)
```

### `state.remember()` 到底保存什么

```python
def remember(
    self,
    actor: str,
    action: str,
    arguments: dict[str, Any] | None = None,
    observation: dict[str, Any] | None = None,
    thought: str = "",
) -> None:
    self.trace.append(
        TraceEvent(
            step=len(self.trace) + 1,
            actor=actor,
            action=action,
            arguments=arguments or {},
            observation=observation or {},
            thought=thought,
        )
    )
```

它保存的是一条“操作历史”，最终追加到 `state.trace`。例如：

```python
state.remember(
    "ReActAgent",
    "read_pr",
    {"number": 18},
    {"number": 18, "title": "Fix expired JWT handling"},
    "先确认 PR 修改范围",
)
```

会产生类似记录：

```python
TraceEvent(
    step=1,
    actor="ReActAgent",
    action="read_pr",
    arguments={"number": 18},
    observation={"number": 18, "title": "Fix expired JWT handling"},
    thought="先确认 PR 修改范围",
)
```

它不会把 `state.pr` 自动改成 PR 数据。真正更新当前证据的是 `apply_observation()`：

```python
def apply_observation(state: EvidenceState, action: str, observation: dict[str, Any]) -> None:
    # 工具返回错误时，不把错误对象当成有效证据写入状态
    if "error" in observation:
        return

    # 根据动作类型，把工具结果写到对应的状态字段
    if action == "read_pr":
        state.pr = observation
    elif action == "read_issue":
        state.issue = observation
    elif action == "read_ci":
        state.ci = observation
    elif action == "read_review_comments":
        state.review_comments = observation.get("comments") or []
    elif action == "read_release_policy":
        state.release_policy = observation
```

所以二者的区别是：

```text
apply_observation -> 更新当前最新证据，供后续判断使用
state.remember     -> 保存过程历史，供调试、展示和审计使用
```

## 3. 工具注册表和白名单

```python
Tool = Callable[..., dict[str, Any]]

TOOLS: dict[str, Tool] = {
    "read_pr": read_pr,
    "read_issue": read_issue,
    "read_ci": read_ci,
    "read_review_comments": read_review_comments,
    "read_release_policy": read_release_policy,
}
```

`TOOLS` 是“动作名称 -> Python 函数”的映射。模型只能返回这些名称，程序再通过：

```python
TOOLS[action](**normalized)
```

找到对应函数并执行。

例如：

```python
normalized = {"number": 18}
TOOLS["read_pr"](**normalized)
```

等价于：

```python
read_pr(number=18)
```

这个白名单很重要，因为模型输出的字符串不能直接当作任意 Python 函数执行。

## 4. 参数标准化

```python
def normalize_arguments(action: str, arguments: Any, state: EvidenceState) -> dict[str, Any]:
    # 只接受字典参数；模型返回列表、字符串或 null 时，按空参数处理
    raw = arguments if isinstance(arguments, dict) else {}

    if action == "read_pr":
        # 没传 number 时使用默认 PR 编号，并统一转换成 int
        return {"number": int(raw.get("number") or PR_DATA["number"])}

    if action == "read_issue":
        # 如果当前 PR 已经带有关联 Issue，则强制使用关联 Issue
        linked_issue = (state.pr or {}).get("linked_issue")
        default = linked_issue or ISSUE_DATA["number"]
        number = default if linked_issue else raw.get("number") or default
        return {"number": int(number)}

    if action in {"read_ci", "read_review_comments"}:
        # CI 和 Review 都以 PR 编号作为输入
        default = (state.pr or {}).get("number") or PR_DATA["number"]
        return {"pr_number": int(raw.get("pr_number") or default)}

    if action == "read_release_policy":
        return {"topic": str(raw.get("topic") or "auth")}

    # 未知动作的参数原样返回；真正的动作白名单检查在 run_tool 中完成
    return raw
```

例如：

```text
输入：action=read_ci，arguments={}
状态：state.pr.number=18
输出：{"pr_number": 18}
```

当前实现使用 `or` 判断“是否传值”，所以 `0`、空字符串和 `False` 会被当成未传入。`int("abc")` 也会直接抛出异常，生产代码应增加明确的参数校验。

## 5. `run_tool()`：一次工具调用的完整生命周期

```python
def run_tool(
    state: EvidenceState,
    actor: str,
    action: str,
    arguments: Any,
    thought: str = "",
) -> dict[str, Any]:
    # 1. 先做白名单检查，防止调用未注册的动作
    if action not in TOOLS:
        observation = {"error": f"工具 {action} 不存在，请从工具白名单里选择。"}

        # 即使工具不存在，也记录这次失败，便于回溯 Agent 行为
        state.remember(actor, "tool_error", {"action": action}, observation, thought)
        return observation

    # 2. 把模型的原始参数补齐并转成工具要求的格式
    normalized = normalize_arguments(action, arguments, state)

    # 3. 查找真实函数并执行，例如 read_pr(number=18)
    observation = TOOLS[action](**normalized)

    # 4. 将成功的结果写入当前证据状态
    apply_observation(state, action, observation)

    # 5. 记录动作、参数、结果和简短原因
    state.remember(actor, action, normalized, observation, thought)

    # 6. 把工具结果交还给调用方或 Agent 循环
    return observation
```

一次成功调用的状态变化如下：

```text
原始参数 {"number": "18"}
    -> normalize_arguments
标准参数 {"number": 18}
    -> TOOLS["read_pr"](**标准参数)
工具结果 {"number": 18, ...}
    -> apply_observation：state.pr = 工具结果
    -> remember：state.trace 增加一条事件
    -> return：把工具结果返回给调用方
```

## 6. 证据是否完整、风险如何确定

```python
def missing_evidence(state: EvidenceState) -> list[str]:
    missing = []
    if state.pr is None:
        missing.append("PR 详情")
    if state.issue is None:
        missing.append("关联 Issue")
    if state.ci is None:
        missing.append("CI 状态")
    if state.review_comments is None:
        missing.append("Review 评论")
    if has_auth_change(state) and state.release_policy is None:
        missing.append("认证模块发布规则")
    return missing
```

`missing_evidence()` 只回答“还缺什么”，并不判断风险。

```python
def deterministic_risks(state: EvidenceState) -> dict[str, list[str]]:
    blockers = []
    warnings = []

    # 失败 CI 是阻塞项
    if state.ci and state.ci.get("status") != "passed":
        blockers.append(f"CI 未通过：{state.ci['failed_test']}")

    # 未解决 Review 评论也是阻塞项
    if unresolved_comments(state):
        blockers.append("仍有未解决 Review 评论")

    # 修改 auth/ 下的文件属于高风险提示
    if has_auth_change(state):
        warnings.append("PR 修改认证模块，属于高风险变更")

    if not state.pr:
        blockers.append("缺少 PR 详情")
    if not state.issue:
        warnings.append("缺少关联 Issue 背景")
    return {"blockers": blockers, "warnings": warnings}
```

这里的规则不依赖模型，因此可以防止模型在 CI 失败时仍然给出“可以合并”的乐观结论。

## 7. 三种 Agent 工作模式

### ReAct

`run_react()` 的模式是“观察 -> 决策 -> 执行 -> 再观察”。每轮模型只决定一个动作：

```python
decision = model.chat_json(
    REACT_PROMPT,
    {"goal": goal, "state": state.model_view(), "available_tools": TOOL_GUIDE},
    "ReAct 决策",
)
```

然后：

```python
action = str(decision.get("action") or "").strip()
thought = str(decision.get("thought") or "")
arguments = decision.get("arguments") or {}
```

如果模型选择 `final`，程序会先检查：

```python
missing = missing_evidence(state)
```

证据不完整时不会允许结束，而是记录 `final_rejected` 并继续循环。

### Plan-and-Execute

`run_plan_and_execute()` 先让 Planner 一次生成计划：

```text
read_pr -> read_issue -> read_ci -> read_review_comments
```

执行每一步后，再让 Re-planner 根据新证据追加步骤，例如发现 PR 修改了 `auth/` 后追加 `read_release_policy`。

### Multi-Agent

`run_multi_agent()` 先加载完整证据，然后分别调用：

```text
IssueAgent      -> 只分析需求背景
PRReviewAgent   -> 只分析代码改动和风险
CIDebugAgent    -> 只分析 CI
SafetyAgent     -> 只分析安全边界
ObserverAgent   -> 检查多个 Agent 是否冲突或缺证据
SynthesisAgent  -> 汇总最终结论
```

这是真正的“多个角色协作”，但当前 `max_steps` 在这个函数中被 `del max_steps` 忽略了，因此它不会限制专用 Agent 的数量。

## 8. 模型调用链路

`ChatModel.chat_json()` 使用标准库 `urllib` 直接调用 OpenAI 兼容的 `/chat/completions` 接口：

```python
payload = {
    "model": self.config.model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ],
    "response_format": {"type": "json_object"},
    "temperature": 0.2,
    "max_tokens": max_tokens,
}
```

请求完成后，程序从：

```python
response_data["choices"][0]["message"]["content"]
```

取出模型回复，再交给 `parse_json_content()` 解析。

`parse_json_content()` 先尝试完整 JSON 解析；如果模型额外输出了说明文字，则截取第一个 `{` 到最后一个 `}` 之间的内容再次尝试。这是兼容性处理，但不能保证所有非标准回复都能被正确修复。

## 9. 最终结论和 Reflection

模型结论会经过：

```python
final = merge_final_with_rules(model_final, state)
```

这里会把模型给出的阻塞项与程序确定性检查出的阻塞项合并，并去重。如果存在阻塞项且决策中没有“暂缓”，程序会强制改成：

```text
暂缓合并
```

如果开启 Reflection：

1. `CriticAgent` 检查最终结论是否忽略失败 CI、未解决 Review 或高风险目录；
2. 如果检查不通过，`WriterAgent` 根据意见重新生成结论；
3. 修正后的结论再次经过 `merge_final_with_rules()`。

## 10. 程序入口

```python
def main(argv: list[str] | None = None) -> int:
    # 解析 --mode、--reflection、--goal 等命令行参数
    args = build_parser().parse_args(argv)

    # 查找 .env 并读取模型配置
    config = load_model_config()
    if args.check_config:
        print_config_status(config)
        return 0

    model = ChatModel(config)
    mode = args.mode or choose_mode_interactively()
    use_reflection = (
        args.reflection == "yes"
        if args.reflection
        else choose_reflection_interactively()
    )

    # 根据模式选择具体执行函数
    runners = {
        "react": run_react,
        "plan": run_plan_and_execute,
        "multi": run_multi_agent,
    }
    result = runners[mode](model, args.goal, args.max_steps)

    # 可选的二次审查
    if use_reflection:
        result = apply_reflection(model, args.goal, result)

    # summary 只打印摘要，json 打印完整结果
    if args.output == "json":
        print_json("运行结果", result)
    else:
        print_run_summary(result)
    return 0
```

入口的完整流程是：

```text
读取命令行参数
  -> 加载 .env
  -> 校验 BASE_URL / MODEL / API_KEY
  -> 创建 ChatModel
  -> 选择 react / plan / multi
  -> 执行 Agent 工作流
  -> 可选 Reflection
  -> 输出 summary 或完整 JSON
```

## 11. 当前代码中建议优先修正的地方

### 11.1 `run_tool()` 缺少异常保护

目前参数转换失败或模型工具函数报错时，异常会直接中断整个流程。可以改成：

```python
def run_tool(state, actor, action, arguments, thought=""):
    if action not in TOOLS:
        observation = {"error": f"工具 {action} 不存在，请从工具白名单里选择。"}
        state.remember(actor, "tool_error", {"action": action}, observation, thought)
        return observation

    try:
        normalized = normalize_arguments(action, arguments, state)
        observation = TOOLS[action](**normalized)
    except (TypeError, ValueError, KeyError, RuntimeError) as exc:
        observation = {"error": f"工具 {action} 执行失败：{exc}"}
        state.remember(actor, "tool_error", arguments if isinstance(arguments, dict) else {}, observation, thought)
        return observation

    apply_observation(state, action, observation)
    state.remember(actor, action, normalized, observation, thought)
    return observation
```

### 11.2 模拟数据被直接返回

```python
return PR_DATA
```

这会返回全局字典本身。如果后续代码修改返回值，也可能修改全局数据。更安全的写法是：

```python
return dict(PR_DATA)
```

嵌套结构较多时应使用 `copy.deepcopy(PR_DATA)`。

### 11.3 `read_issue` 会强制忽略调用参数

当 `state.pr["linked_issue"]` 存在时，即使调用方传入 `number=999`，也会读取关联 Issue。这个行为可能是设计，也可能让人困惑，建议加注释或改成明确的参数优先级。

### 11.4 环境变量命名需要统一

代码同时接受：

```text
BASE_URL / LLM_BASE_URL
MODEL    / LLM_MODEL
API_KEY  / LLM_API_KEY
```

建议项目最终只保留一套命名，或者在 `.env.example` 中明确说明优先级。

### 11.5 输出文字仍然写着 DeepSeek

`print_run_summary()` 中的：

```python
print("=== DeepSeek 实际返回摘要 ===")
```

如果现在使用 Qwen，应改成：

```python
print("=== 模型实际返回摘要 ===")
```

这样不会把模型品牌写死。

### 11.6 `response_format` 需要确认 Qwen 接口支持情况

代码要求模型返回：

```python
"response_format": {"type": "json_object"}
```

虽然很多 OpenAI 兼容接口支持这个字段，但具体取决于服务商和模型。若 Qwen 接口报参数不支持，需要删除该字段，改由 Prompt 强制 JSON，再使用 `parse_json_content()` 做解析。

## 12. 一次实际运行的简化示例

假设目标是：

```text
判断 PR #18 是否可以合并
```

ReAct 可能经历：

```text
第 1 步：state 为空，模型选择 read_pr
第 2 步：读取 PR #18，发现修改了 auth/，并关联 Issue #12
第 3 步：读取 Issue #12
第 4 步：读取 PR #18 的 CI，发现失败
第 5 步：读取 Review，发现有未解决评论
第 6 步：读取 auth 发布规则
第 7 步：证据齐全，生成最终结论
```

最终因为：

```text
CI 未通过
Review 评论未解决
修改了认证模块
```

`merge_final_with_rules()` 会保证结论至少包含：

```json
{
  "decision": "暂缓合并",
  "blockers": [
    "CI 未通过：test_expired_jwt_returns_401",
    "仍有未解决 Review 评论"
  ],
  "warnings": [
    "PR 修改认证模块，属于高风险变更"
  ]
}
```

