# Agent libOS 预上线子系统评审与硬化报告

日期：2026-07-10

> **历史、commit-bound 快照。** 本报告描述并验证的是 Git commit
> `3681d80b86e754ffc3ac6beae63ac76f1454432c` 当时的实现、命令面、测试数量与
> 本机环境。它不是当前分支或 dirty working tree 的实时发布状态；后续提交的
> 命令数、测试数、benchmark 计数和剩余环境 gate 不应回填或从本报告推断。
> 当前行为以 [README](../README.md)、实现文档和
> [支持与验证矩阵](support_matrix.md) 为准。

## 结论摘要

本轮评审覆盖运行时、Capability、Object Memory、存储、Provider
Substrate、文件系统、Shell、Clock、Human I/O、LLM 执行器、Checkpoint、
Image、Skill/JIT、JSON-RPC、MCP、PTY Runtime Module、GUI/API、资源管理、
benchmark 与文档。评审不只检查正常路径，还检查了权限消耗、异常窗口、
事务回滚、并发竞争、宿主崩溃、外部副作用证据和文档与实现的一致性。

评审前最主要的系统性风险不是单一越权入口，而是多个边界在
“权限已检查、provider 已开始、状态尚未持久化”之间存在不一致的故障窗口。
这些窗口会造成一次性权限错误退款或重复消费、外部副作用缺少可恢复证据、
数据库与内存 payload 分叉、恢复/分叉时复活旧权限，以及进程异常退出后遗留
不受控子进程。本轮已把这些路径统一为可检查的状态机，并为关键不变量增加
确定性回归。

截至本报告完成时，没有已知、可稳定复现且尚未处理的高风险功能回归。
完整确定性 Python 矩阵为 1187 passed、5 skipped；跳过项均是未配置的真实
PostgreSQL 或非当前平台分支，不是静默忽略的失败。仍需在发布前补齐真实
PostgreSQL、真实 Windows 和真实 LLM 的环境级验证，详见“验证边界”。

## 总体架构判断

项目的正确安全边界仍然是：

```text
process identity + Capability + primitive + provider containment + audit
```

Tool、Skill、JIT、Image、Checkpoint 和 GUI 是可见性、编排或人体交互表面，
不应自行授予资源权限。本轮修改保留了这一设计，并进一步落实了三个原则：

1. 任何可观察或可变更外部状态的 provider 边界，在首次外部观察前先持久化
   pending intent。
2. 有限次数权限先 reservation；仅 provider 明确证明未开始时才允许原子退款，
   一旦发生观察、写入或结果不确定，就消费权限并保留 UNKNOWN 证据。
3. 关系行、payload、权限消耗、event 与 audit 要么在同一事务提交，要么在
   无法原子化的外部边界上保留可恢复、不可重放的持久状态。

## P1 修复复核

在首次全面评审后，所有 P1 项均已按对应不变量修复并增加回归：

- 真实 LLM 的 `AsyncOpenAI` transport 改为每次请求创建并关闭；显式注入的
  host/test transport 则保留到 `LLMClient`/profile registry shutdown，注入方须
  保证其 event-loop 生命周期兼容。
- public single-step 统一进入 scheduler quantum lock；core process exec、exit 与
  父进程 wake、message insert/evidence/wake 的 store transition 均以事务提交。
  高层 image boot 在随后阶段失败时使用补偿恢复，不宣称跨 host/package 工作的
  单一 SQL 事务。
- Capability issue 的 row、process attachment、有限签发权消费、event/audit 原子
  提交；Checkpoint inspect/diff/replay 与跨 actor ObjectTask get/list/wait/cancel
  会按各自操作粒度实际消费有限次数权限。
- Human terminal read/write 只在锁内 claim 和提交状态，阻塞 provider I/O 位于锁外；
  Human API 列表 pending-first，GUI snapshot 再施加通用 collection 上限并报告截断。
- 精确内置 Git 只读 allowlist 关闭 optional locks、fsmonitor 和 external diff；SQLite 主库、
  lease、journal、WAL 与 SHM 在支持 `O_NOFOLLOW`/`fchmod` 的 POSIX 安全路径上
  创建或收紧为 owner-only `0600`。
- runtime-safety CLI 在执行前写入 `metadata.json`，聚合器以其中的 task×runner
  笛卡尔积作为完成性清单；
  缺失或意外 result pair 都会使指标失效，不再把部分结果误报为完整成功。

## 子系统评审结果

| 子系统 | 发现的设计或实现问题 | 已完成的修复 | 对整体系统的影响 |
| --- | --- | --- | --- |
| SQLite/PostgreSQL 存储 | SQL 回滚与进程内 Object payload 可能分叉；提交/回滚失败后连接仍可能继续使用；旧 schema 重建、运行时 lease 隔离和 SQLite 文件权限不够严格 | transaction 同步快照 payload；回滚失败时 poison/关闭 store；schema migration 可恢复；安全 POSIX 路径上的 SQLite 主库、lease、journal/WAL/SHM 创建或收紧为 `0600`，fallback 依赖平台 ACL；PostgreSQL advisory key 纳入 database/schema | 防止已回滚对象在内存中继续可见、双 runtime 误共用 store、半迁移 schema 和本机敏感数据旁路读取 |
| 外部副作用模型 | 过去只有完成后的 effect record，provider 开始后若 event/audit/classifier 失败会丢失证据 | 引入 `pending/finalized` effect state；intent 与最终 record 使用同一 effect id；finalize 采用身份绑定 CAS；PENS 才能 abandon | 崩溃恢复与 benchmark 不再把“没有证据”误当作“没有发生”，外部副作用默认保守收口 |
| Capability 与一次性权限 | 多个 primitive 在 provider 前直接 consume，导致明确未执行时无法退款；组合权限与 capability issue 可能部分提交 | 统一 reserve/commit/restore；同一 capability 去重；组合权限和 issue row/attachment/evidence 在一个事务提交；恢复/分叉使旧 reservation 失效 | 消除一次性权限丢失、重复使用与跨边界部分提交 |
| Filesystem 与 cwd | cwd 验证可在权限前通过 `state` 或 `Path.resolve()` 观察存在性、类型和 symlink 目标；写入审批曾提前探测 target | `resolve` 改为纯 lexical；真实 containment/symlink 检查延后到授权后；cwd 必须具备目录 READ；state probe 纳入 intent 与资源计费；审批只显示调用方声明和内容摘要 | 关闭目录存在性/符号链接 oracle，并使 shell、child、fork、PTY 的 cwd 语义一致 |
| Shell 与 Clock | timeout、取消和 provider 异常的权限/副作用结果不一致；只读 Git 可执行 repo-configured external diff/fsmonitor；psutil 原生 `PermissionError` 会遮蔽 wall-time 和清理语义 | Shell/Clock 均使用结构化 intent；Git 关闭 optional lock、fsmonitor 和 external diff；wall/timeout 依赖 monotonic 与进程组；CPU/RSS 无法完整采样时 fail closed；kill 路径按进程组、tree、direct child 回落 | 已覆盖的只读 Git argv 不再执行仓库 external diff/fsmonitor；资源限制在受限宿主仍可预测，未知执行不会错误退款，后台子进程不会因采样错误失控 |
| Human I/O 与审批 | 并发响应、无类型 answer、隐式策略、terminal retry 可能重复提示；terminal lock 覆盖阻塞 I/O；有界历史可能遮蔽 pending；敏感文本可能进入 metadata | permission 必须显式选择策略；terminal read/write 锁内 claim、锁外 I/O；列表 pending-first；每个 run 使用不可变 Human context；pending intent 只保存长度、hash、purpose 和 error type | 防止审批 ABA、跨 run 策略串扰、exit 饥饿、pending 被历史记录挤出 GUI snapshot、重复人体副作用和敏感文本持久化 |
| Object Memory | owner/version 检查与 payload 更新之间存在竞态；finalizer 在 SQL 事务内可能造成不可回滚外部 close；event/audit 失败可留下孤立对象或 link | ownership transition lock；LIVE/owner/version CAS；finalizer 位于 SQL 事务外但在所有权锁内；namespace/object/link/update/delete 与 capability/event/audit 原子提交；listing 消费实际使用的有限可见性权限 | 防止 stale update 复活对象、转移/删除误报成功、PTY close 重复，以及有限 READ 无限枚举 metadata |
| Process、Scheduler 与 ResourceManager | public single-step 可绕过 scheduler lock/charge；core exec、exit/parent wake 和 message wake 可能半提交；清理一个 PID 失败可阻断其他进程 | single-step 统一进入 quantum lock；core exec/message/exit+wake 的 store transition 事务化；高层 image boot 失败使用补偿恢复；per-PID best-effort finalize；run-local Human context；shutdown 可重试；资源 charge 对祖先链原子发布 | 防止同 PID 重入、资源计费旁路、半终止/半唤醒，并减少孤儿进程和半关闭 runtime；不把补偿式 image boot 误述为单事务 |
| LLM 执行器与上下文 | 真实 async SDK client 可跨已关闭 event loop 复用；pending action 可能在 effect 后、clear 前被重放；把全量本地快照与 Responses provider state 叠加会重复历史 | 真实 client request-scoped 创建/关闭；每代 wait 使用唯一 resume token 和 pending→resuming CAS；AgentProcess 禁用 `previous_response_id` 并记录原因；Runtime 只记录 scope fingerprint 供观测和未来协议使用，低层 client 并不执行该隔离检查 | 防止跨 loop transport 崩溃、非幂等动作重复执行和全量历史双重回放，同时不把尚未启用的 fingerprint 误报为低层租户隔离防线 |
| Checkpoint、Image、ObjectTask 与 fork | read/cancel 有限权限可被重复使用；snapshot/head/capability/event/audit 可能分步提交；fork 发布前权限可被撤销 | inspect/diff/replay 和跨 actor ObjectTask get/list/wait/cancel 按操作粒度消费有限权限；checkpoint create 同事务；restore/fork scope lock；publish 重验 authority；不克隆有限能力；JIT identity 重映射 | 防止有限权限无限读取/取消、撤销后能力复活、半发布 fork 和跨进程 JIT identity 共享 |
| Deno/JIT 与 Skill | 宿主被 SIGKILL 时，纯 parent-side monitor 无法保证 Deno 退出；SWE edit 可能在截断源码上写回 | POSIX death pipe + 独立进程组；Windows `KILL_ON_JOB_CLOSE` Job Object gate；containment 建立失败即拒绝启动；SWE edit 在 source 截断时拒绝写入 | 防止宿主异常退出留下无限 CPU/RSS orphan，避免截断输入破坏工作区 |
| JSON-RPC | endpoint metadata 可在 registry 权限前读取；DNS 发生在 intent 前；registry row、stale grants、event/audit 可能分叉 | 先鉴权再加载 metadata；DNS 前持久化 intent 和一次性 method reservation；DNS 后任何不确定失败保留信息流；registry mutation 原子化 | 关闭 endpoint existence oracle，避免 DNS/transport 已观察却退款，防止注销后 stale 权限继续存在 |
| MCP | live validation、list/call 与 stdio spawn 需要多项权限，旧路径可能部分消费或在 DNS 前无证据；registry 同样存在 metadata oracle | 主 tool、server、process:spawn、精确 stdio EXECUTE 组合预约；HTTP DNS 纳入 intent；call 与 live validation 共用一个 effect 边界；registry 先鉴权并原子提交 | 保证远程/本地 MCP 在失败窗口的权限与证据一致，并阻止通过 registry 发现未授权 server |
| PTY Runtime Module | write/resize/close 直接消费有限 Object 权限；自动退出先读 exit code/close 再记录 effect；reader 与 monitor 相互阻塞 | mutation 使用 reserve/commit/restore；auto-exit 在首次 provider 调用前创建 close intent；list 消费有限 READ；POSIX reader/资源 monitor 独立且 wall charge 先于可失败 sampler；Windows 无该 supervisor，带 `SubprocessLimits` 时启动前拒绝 | 防止交互式终端权限异常退款、自动退出副作用丢失和 POSIX blocked reader 绕过预算；不把 Windows ConPTY I/O 误述为资源监管 |
| ToolBroker 与 workflow | 全局 name fallback 可解析其他进程的 ephemeral JIT；未知 workflow 名称被当作真实 tool id 返回 | 移除跨进程 ephemeral fallback；缺失工具返回结构化 table denial；workflow 只有成功解析时才暴露 tool id | 关闭跨进程 JIT 可见性通道，API 不再把请求字符串伪装成已注册 identity |
| GUI/API | Human 请求类型与策略在 UI/API 中表达不足；有界历史可遮蔽新 pending；shutdown 与并发 runtime user 的生命周期不完整 | typed Human request card 与显式策略；Human list pending-first，snapshot 在通用上限下保序并报告截断；API 严格校验；GUI shutdown drain/retry；client 类型、测试、i18n 与样式同步 | 降低误审批和 API 模糊输入，避免历史记录挤掉待处理请求或关闭时访问已释放 store；不承诺单个 snapshot 可容纳超过全局 collection 上限的 pending |
| Benchmark | 旧 runner 可由 `result.ok` 推断副作用；match 规则和分母不够严格；部分 result 文件可被当作完整 run | schema v1 evidence-first oracle；exact/prefix/glob fail closed；CLI 预写 metadata 后执行 task×runner 完成性校验；未知/缺失/意外证据使 run invalid；显式分母字段 | 指标不再掩盖无证据或缺失任务，27 个任务可重复地产生完整 effect certificate |

## 文档评审

实现与文档已同步检查。重点修订如下：

- `README.md` 与 `docs/architecture.md` 明确 primitive/provider/effect-intent
  边界，以及 JIT supervisor 的宿主生命周期保证。
- `docs/runtime_model.md` 说明 cwd READ、Human terminal intent、shutdown、
  durable wait generation 和 process lifecycle。
- `docs/capabilities.md` 说明 reservation、PENS、授权前禁止状态探测和组合权限。
- 新增 `docs/storage.md`，记录 rollback/poison、payload durability、migration
  recovery 与 runtime lease。
- `docs/object_memory.md`、`docs/checkpoints.md`、`docs/modules.md` 分别同步
  owner lock/finalizer、fork publish revalidation 和 PTY/JIT trust hash。
- `docs/jsonrpc.md` 与 `docs/mcp.md` 说明 registry metadata gate、DNS intent、
  composite reservation 与事务原子性。
- `docs/gui.md`、`docs/cli.md`、`docs/benchmark.md` 与实际 API、25 个顶层 CLI
  命令和 schema-v1 指标一致。

Markdown 相对链接检查通过；CLI 源码抽取的 25 个顶层命令与文档一致；
PTY manifest/source hash 与 `config.yaml` 的信任项一致。

## 验证证据

### 自动化矩阵

| 验证项 | 结果 |
| --- | --- |
| 全部确定性 Python lanes（完整本机环境，含真实 Deno） | 1187 passed, 5 skipped |
| 覆盖 lanes | unit、runtime、security、self-evolution、providers、benchmark |
| GUI Vitest | 19 files, 65 tests passed |
| GUI typecheck/build | web + Electron TypeScript 与 Vite production build 通过 |
| invariant manifest | 42 invariants / 1197 collected pytest nodes |
| Python compileall | 通过 |
| `git diff --check` | 通过 |

五个全矩阵 skip 分别是三个需要 `AGENT_LIBOS_POSTGRES_DSN` 与
`--run-postgres` 的真实 PostgreSQL 节点、一个 Windows PTY 专用分支和一个
当前平台不使用的 filesystem fallback-open 分支。

### 27 任务确定性 runtime-safety certificate

- 结果有效：27/27。
- task success：27/27。
- safety pass：27/27。
- 未授权且明确执行的副作用：0/22。
- 未知 classification/outcome：0。
- false denial：0/22（0%）。
- audit completeness：100%。
- normalized effect records：47。
- tool calls：73；primitive calls：16。

该结果是实现与证据链的确定性验证，不代表完整论文评测，也不等价于形式化
安全证明。

## 验证边界与发布前建议

以下项目是环境覆盖缺口，不是本轮已复现缺陷：

1. 当前环境没有配置 `AGENT_LIBOS_POSTGRES_DSN`，因此 PostgreSQL 只完成了
   shared-contract、SQL 生成、lease key 与静态/单元验证。发布 PostgreSQL
   backend 前应在真实实例上运行 `--run-postgres`。
2. 当前主机为 macOS。Deno 的 Windows Job Object 分支和 PTY 的 ConPTY
   分支有 mock/branch 测试，但当前 PTY provider 本身没有 Job Object 或
   CPU/RSS/wall 监管。发布 Windows 支持前仍应在真实 Windows runner 上执行
   GUI、PTY、Shell 和 Deno parent-death 集成测试。
3. 真实 LLM 路径按项目策略保持 opt-in，本轮没有消费 token。发布 provider
   配置前应分别对 Responses 与 Chat Completions 做最小真实 smoke，并保持
   `store=false` 与 `responses_previous_response_id=false` 的默认无状态
   provider 姿态。它只限制 provider 侧状态；默认
   `llm.persist_full_io=true` 仍会在 RuntimeStore 保留完整 I/O，因此不是
   端到端隐私保证。
4. GUI bundle 缺少 `npm`，因此本轮使用 package.json 等价命令完成 Vitest、
   两套 TypeScript 配置和 production build；CI 仍应通过标准 `npm ci` 后运行
   官方 GUI lane。

建议将真实 PostgreSQL、Windows 和真实 LLM smoke 作为发布 gate，而不是把
它们并入默认 token-free 确定性矩阵。

## 最终判断

从当前实现与可运行证据看，项目已从“多个子系统各自处理异常”收敛为统一的
授权、reservation、外部副作用 intent、事务和审计模型。修复显著降低了
越权 oracle、一次性权限错配、不可恢复半提交、动作重放和孤儿进程风险。
在补齐上述三个真实环境 gate 前，适合继续作为预发布候选和论文 artifact；
补齐后可进入正式 release candidate 阶段。
