# Agent libOS Historical Design Archive

> **ARCHIVE — NOT A CURRENT CONTRACT.** This historical design archive preserves
> early design intent and future directions; it is not the source of truth for
> the current implementation. Sections below intentionally contain planned or
> superseded capability and approval flows, commands, security properties,
> checkpoint behavior, JIT languages and sandboxes, storage choices, and payload
> retention rules. Do not use them to infer current behavior or guarantees. Use
> [README.md](README.md) and the current `docs/` references for current behavior,
> and [docs/invariants.md](docs/invariants.md) for the invariant-to-test map.

## 0. 文档目的

本文档用于指导团队实现一个以 **Agent Process** 为核心抽象的 Agent libOS 框架。该框架不以传统 LLM 聊天界面为中心，而是将
Agent 建模为长期运行、可调度、可中断、可扩展、可审计的执行主体。

系统目标是提供一组 Agent-native 的运行时原语，使 Agent 能够：

- 作为进程长期运行；
- fork 子 Agent；
- exec 到新的执行镜像；
- 动态加载 skills；
- 在受控条件下 JIT 生成并注册新工具；
- 使用对象化内存，而不是字节寻址内存或文件系统目录树；
- 将人类建模为可访问、可授权、可发送中断的外部对象；
- 通过 capability、安全沙箱、审计日志和 checkpoint/rollback 机制控制风险。

本文档面向开发团队，重点说明系统边界、核心模块、数据模型、API、执行语义、安全约束和实现路线。

---

## 1. 设计原则

### 1.1 Agent 是运行主体，不是对话响应器

传统 Chatbot 架构通常是：

```text
Human -> Chat Loop -> Agent -> Tools
```

本系统采用 Agent-centric 架构：

```text
Agent Process -> Tools / Skills / Memory / HumanObject / Primitives
```

人类不是每一步主循环的驱动者，而是 Agent libOS 中的一类特殊 primitive：Agent 可以主动访问人，人也可以通过中断机制影响 Agent。HumanObject 负责请求队列、审批、唤醒、审计与 capability 语义；具体终端或 UI 的读写落在 Resource Provider Substrate 的 `HumanProvider` 上。

### 1.2 Agent Process 而非 Workflow Thread

本框架的核心抽象不是 workflow run，而是 Agent Process。

Agent Process 应具有：

- 稳定身份：`pid`；
- 父子关系：`parent_pid`；
- 执行镜像：`AgentImage`；
- 当前目标：`Goal`；
- 当前状态：`ProcessState`；
- 对象化工作集：`MemoryView`；
- 能力集合：`CapabilitySet`；
- 已加载技能：`SkillSet`；
- 工具句柄表：`ToolHandleTable`；
- 事件队列：`EventQueue`；
- 审计日志：`AuditLog`；
- checkpoint/rollback 能力。

### 1.3 Object Memory，而不是 byte-addressed memory

Agent 的内存不应模拟虚拟地址空间。Agent 处理的是计划、证据、观察、工具结果、代码补丁、人类决策、任务状态和技能对象，而不是裸字节。

内存应实现为：

```text
Typed Object Store + Object Graph + Capability Handles + Memory Views + Context Materialization
```

文件系统只是外部对象适配器之一，不是整个系统的根抽象。

### 1.4 Everything is object/event/capability，不是 everything is file

统一性来自三件事：

- object：所有可引用实体都有对象身份；
- event：所有状态变化和异步交互都通过事件传播；
- capability：所有访问、修改、副作用和扩展都受 capability 管控。

### 1.5 Agent 可以自扩展，但不能自授权

Agent 可以提出：

- fork 子 Agent；
- exec 到新镜像；
- load skill；
- propose JIT tool。

但它不能绕过 runtime 的 capability manager、tool broker、sandbox、policy checker 和 human approval。

核心原则：

> Agent may propose capability expansion, but the runtime decides whether to grant it.

### 1.6 所有外部副作用必须可追踪

包括：

- 文件写入；
- shell 命令；
- 网络请求；
- API 调用；
- 发邮件；
- 创建日程；
- 写数据库；
- 注册工具；
- 加载技能；
- 修改长期记忆；
- 人类授权；
- capability grant/revoke。

所有这些都必须进入 audit log。

---

## 2. 总体架构

### 2.1 分层结构

```text
+------------------------------------------------------------+
| Agent Applications / Personalities                         |
| CodingAgent / ResearchAgent / EDAAgent / TutorAgent        |
+------------------------------------------------------------+
| Skills / Tools Layer                                       |
| LLM-facing actions, skills, tool bundles, workflows        |
| Wrap and compose libOS primitives into usable affordances   |
+------------------------------------------------------------+
| Agent LibOS                                                |
| Process API / Object Memory API / Event API / Human API    |
| Skill Loader / Tool Broker Interface / Context Materializer|
+------------------------------------------------------------+
| Agent Kernel ABI                                           |
| Process / Event / Capability / Object / Checkpoint / Audit |
+------------------------------------------------------------+
| Host Runtime                                               |
| Container Sandbox / LLM API / DB / Queue / Object Store    |
| External Service Adapters / Human UI / Policy Engine       |
+------------------------------------------------------------+
```

**Skills / Tools Layer** 是面向 LLM 暴露的 action surface，负责把 libOS 的低层原语组合、约束和文档化，使模型能够以稳定、可理解、可验证的方式使用系统能力。

类比传统系统：

```text
Application
  -> libc / language runtime / standard library
  -> syscall ABI
  -> kernel
```

在本系统中对应为：

```text
Agent Personality
  -> Skills / Tools Layer
  -> Agent LibOS API
  -> Agent Kernel ABI / Host Runtime
```

### 2.2 核心组件

```text
AgentRuntime
 ├── ProcessManager
 ├── Scheduler
 ├── EventBus
 ├── CapabilityManager
 ├── ObjectMemoryManager
 ├── ContextMaterializer
 ├── SkillLoader
 ├── ToolBrokerInterface
 ├── HumanObjectManager
 ├── CheckpointManager
 ├── Primitive Managers
 ├── AuditManager
 └── PolicyEngine

SkillsToolsLayer
 ├── SkillRegistry
 ├── ToolRegistry
 ├── ToolBundleManager
 ├── ActionSchemaCompiler
 ├── Prompt/Instruction Packager
 ├── Workflow Macro Library
 ├── Tool Selection Metadata
 └── Adapter Library
```

需要区分：

- `AgentRuntime` 提供底层可执行、安全、状态和审计能力；
- `SkillsToolsLayer` 提供 LLM 可用的高层动作、工具描述、技能说明、组合宏和领域适配器；
- `Agent Applications / Personalities` 选择、配置和约束某一类任务所需的 skills/tools。

### 2.3 Skills / Tools Layer 的职责

Skills / Tools Layer 位于 Agent personality 与 Agent LibOS 之间，承担类似 libc、语言运行时和标准库的职责。

它的主要职责包括：

1. **封装底层原语**

   将低层 libOS 调用包装为 LLM 可理解的动作。

   例如：

   ```text
   spawn_log_analysis_worker(log_object)
     = create_view(log_object, read_only)
       + fork(mode=WORKER)
       + wait(child)
       + merge_view(child_result)
   ```

2. **定义 LLM-facing action schema**

   每个暴露给模型的 tool/action 都必须有清晰的：

    - 名称；
    - 使用场景；
    - 输入 schema；
    - 输出 schema；
    - 权限需求；
    - 副作用说明；
    - 失败模式；
    - 示例。

3. **组合常用工作流**

   将多步 libOS 操作组合为稳定的 macro-action。

   例如：

   ```text
   run_tests_and_summarize(test_command)
   inspect_process_status(pid)
   rollback_to_last_safe_checkpoint()
   ```

4. **隔离模型与底层复杂性**

   Agent 不应直接调用 `capability.grant`、`checkpoint.restore`、`memory.merge_view` 等危险或复杂原语，尽管原语内部有
   PolicyEngine 检查。此类能力应通过受限 tool 或 skill 暴露。

5. **支持领域专用能力包**

   不同 Agent personality 可以加载不同 tool bundles：

   ```text
   Coding Tool Bundle
     - read_repo_structure
     - analyze_test_failure
     - propose_patch
     - run_tests
     - request_patch_approval

   Research Tool Bundle
     - search_papers
     - extract_claims
     - compare_methods
     - build_bibliography

   EDA Tool Bundle
     - run_cli_command
     - parse_waveform
     - inspect_timing_report
     - request_design_decision
   ```

6. **作为 JIT tool 的落点**

   Agent 生成的新工具不应直接进入 libOS，而应进入 Skills / Tools Layer 的 registry。ToolBroker 负责验证和注册，Skills /
   Tools Layer 负责将其包装成 LLM 可用 action。

### 2.4 关键数据流

#### 2.4.1 Agent 执行一轮

```text
ProcessManager selects runnable AgentProcess
  -> Scheduler grants execution budget
  -> ContextMaterializer materializes selected MemoryView
  -> LLM/Planner produces action
  -> Runtime validates action through PolicyEngine
  -> Action dispatched to Tool/Skill/Memory/Human/Event subsystem
  -> Result written as Object + Event + Audit record
  -> Process state updated
```

#### 2.4.2 Agent 请求人类授权

```text
AgentProcess proposes high-risk action
  -> CapabilityManager detects missing capability
  -> HumanObjectManager creates HumanRequestObject
  -> EventBus emits human_query event
  -> Process moves to WAITING_HUMAN or continues with alternative path
  -> Human replies approve/reject/edit
  -> EventBus emits human_response event
  -> CapabilityManager grants/rejects capability
  -> Process resumes or replans
```

#### 2.4.3 Agent JIT 生成工具

```text
AgentProcess proposes ToolCandidateObject
  -> ToolBroker builds in sandbox
  -> Static analysis + unit tests + permission analysis
  -> PolicyEngine evaluates risk
  -> Optional HumanObject approval
  -> ToolRegistry signs and registers executable tool
  -> Skills / Tools Layer wraps tool as LLM-facing action
  -> Agent receives ToolHandle with attenuated capabilities
```

---

## 3. 核心抽象

## 3.1 AgentProcess

### 3.1.1 定义

AgentProcess 是系统中可调度的长期运行主体。

```python
@dataclass
class AgentProcess:
    pid: PID
    parent_pid: PID | None
    image: AgentImageRef
    state: ProcessState
    goal: ObjectHandle
    memory_view: MemoryView
    capabilities: CapabilitySet
    loaded_skills: dict[SkillID, SkillHandle]
    tool_table: dict[ToolID, ToolHandle]
    event_cursor: EventCursor
    checkpoint_head: CheckpointID | None
    status: ProcessStatus
    message_queue: ProcessMessageQueue
    resource_budget: ResourceBudget
    created_at: Timestamp
    updated_at: Timestamp
```

### 3.1.2 ProcessStatus

```python
class ProcessStatus(Enum):
    CREATED = "created"
    RUNNABLE = "runnable"
    RUNNING = "running"
    WAITING_EVENT = "waiting_event"
    WAITING_TOOL = "waiting_tool"
    WAITING_HUMAN = "waiting_human"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    EXITED = "exited"
    FAILED = "failed"
    KILLED = "killed"
```

### 3.1.3 Process 原语

```python
class ProcessAPI:
    def fork(
            self,
            parent: PID,
            goal: ObjectHandle,
            memory_view: MemoryViewSpec,
            capabilities: CapabilitySpec,
            image: AgentImageRef | None = None,
            mode: ForkMode = ForkMode.RESTRICTED,
    ) -> PID: ...

    def exec(
            self,
            pid: PID,
            image: AgentImageRef,
            args: dict,
            preserve_memory: bool = True,
            preserve_capabilities: bool = False,
    ) -> None: ...

    def wait(self, pid: PID, child: PID, timeout: Duration | None = None) -> ProcessResult: ...

    def signal(self, target: PID, signal: ProcessSignal, payload: dict | None = None) -> None: ...

    def send_message(
            self,
            sender: PID,
            recipient: PID,
            kind: Literal["normal", "interrupt"],
            subject: str,
            body: str,
            payload: dict | None = None,
    ) -> ProcessMessageID: ...

    def read_messages(
            self,
            pid: PID,
            include_acked: bool = False,
            ack: bool = True,
    ) -> list[ProcessMessage]: ...

    def pause(self, pid: PID, reason: str) -> None: ...

    def resume(self, pid: PID) -> None: ...

    def cancel(self, pid: PID, reason: str) -> None: ...

    def exit(self, pid: PID, result: ObjectHandle | None = None) -> None: ...
```

### 3.1.4 fork 语义

支持四种 fork 模式：

```python
class ForkMode(Enum):
    COPY = "copy"
    RESTRICTED = "restricted"
    SPECULATIVE = "speculative"
    WORKER = "worker"
```

#### COPY

子进程继承父进程较完整的 memory view，但 capability 默认仍需 attenuate。

用于：同级分支探索、可控可信任务。

#### RESTRICTED

子进程只获得显式指定对象和最小权限。

用于：默认安全模式。

#### SPECULATIVE

子进程用于探索，不允许产生外部副作用，结果需 merge 才能进入主线。

用于：多方案并行、代码修复尝试、计划搜索。

#### WORKER

子进程只执行封闭任务，无长期记忆写入权限。

用于：日志分析、测试运行、局部检索、格式转换。

### 3.1.5 exec 语义

`exec` 替换进程执行镜像，但不必替换全部状态。

执行镜像包括：

- LLM context；
- system instruction；
- planner strategy；
- default skills；
- default tools；
- context materialization policy；
- safety policy profile；
- output/action protocol。

`exec` 必须经过：

- image signature check；
- policy compatibility check；
- capability preservation check；
- state migration check；
- audit record。

默认规则：

```text
exec 不自动提升 capability。
exec 后 capability 只能保持不变或收缩。
需要新 capability 时必须显式 request。
```

---

## 3.2 AgentImage

### 3.2.1 定义

AgentImage 是 Agent Process 的执行镜像，类似可执行文件或容器镜像。

```python
@dataclass(frozen=True)
class AgentImage:
    image_id: AgentImageID
    name: str
    version: str
    system_prompt: str
    planner: PlannerSpec
    action_schema: ActionSchema
    default_skills: list[SkillRef]
    default_tools: list[ToolRef]
    context_policy: ContextPolicy
    safety_profile: SafetyProfile
    required_capabilities: list[CapabilityRequirement]
    metadata: dict
    signature: Signature | None
```

### 3.2.2 镜像类型

推荐内置以下镜像：

- `base-agent`：通用任务执行；
- `coding-agent`：软件工程任务；
- `research-agent`：文献调研；
- `eda-agent`：EDA/CLI 操作；
- `toolmaker-agent`：工具生成与测试；
- `review-agent`：审查、验证、安全分析；
- `summarizer-agent`：摘要压缩；
- `human-coordinator-agent`：人类交互协调。

---

## 3.3 Object Memory

## 3.3.1 设计目标

Object Memory 是 Agent libOS 的核心内存系统。

它应支持：

- typed objects；
- object graph；
- versioning；
- provenance；
- capability-protected access；
- snapshots；
- memory views；
- fork/merge；
- semantic query；
- context materialization。

### 3.3.2 AgentObject

```python
@dataclass(frozen=True)
class AgentObject:
    oid: OID
    type: ObjectType
    schema_version: str
    payload: bytes | dict | str
    metadata: ObjectMetadata
    provenance: Provenance
    version: int
    immutable: bool
    created_by: PID | SystemActor
    created_at: Timestamp
    updated_at: Timestamp
```

### 3.3.3 ObjectMetadata

```python
@dataclass
class ObjectMetadata:
    title: str | None
    summary: str | None
    tags: list[str]
    mime_type: str | None
    token_estimate: int | None
    embedding_refs: list[EmbeddingRef]
    indexes: list[IndexRef]
    sensitivity: SensitivityLevel
    retention_policy: RetentionPolicy
```

### 3.3.4 常见 ObjectType

```python
class ObjectType(Enum):
    TASK = "task"
    GOAL = "goal"
    PLAN = "plan"
    STEP = "step"
    CONSTRAINT = "constraint"
    MESSAGE = "message"
    HUMAN_DECISION = "human_decision"
    HUMAN_REQUEST = "human_request"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    ERROR_TRACE = "error_trace"
    CODE_PATCH = "code_patch"
    TEST_RESULT = "test_result"
    EVIDENCE = "evidence"
    CLAIM = "claim"
    SUMMARY = "summary"
    SKILL = "skill"
    TOOL_SPEC = "tool_spec"
    TOOL_CANDIDATE = "tool_candidate"
    TOOL_ARTIFACT = "tool_artifact"
    CHECKPOINT = "checkpoint"
    PROCESS_STATE = "process_state"
    EXTERNAL_REF = "external_ref"
    ARTIFACT = "artifact"
```

### 3.3.5 ObjectHandle

OID 不代表访问权限。Agent Process 必须通过 capability handle 访问对象。

```python
@dataclass(frozen=True)
class ObjectHandle:
    oid: OID
    rights: set[ObjectRight]
    capability_id: CapabilityID
    expires_at: Timestamp | None
```

```python
class ObjectRight(Enum):
    READ = "read"
    WRITE = "write"
    LINK = "link"
    DIFF = "diff"
    MATERIALIZE = "materialize"
    DELETE = "delete"
    GRANT = "grant"
```

### 3.3.6 Object Graph

对象之间通过 typed links 形成图。

```python
@dataclass(frozen=True)
class ObjectLink:
    src: OID
    relation: RelationType
    dst: OID
    metadata: dict
    created_by: PID | SystemActor
    created_at: Timestamp
```

常见关系：

```python
class RelationType(Enum):
    HAS_PLAN = "has_plan"
    HAS_STEP = "has_step"
    CONSTRAINED_BY = "constrained_by"
    SUPPORTED_BY = "supported_by"
    PRODUCED = "produced"
    EVALUATED_BY = "evaluated_by"
    DERIVED_FROM = "derived_from"
    SUMMARIZES = "summarizes"
    REFERENCES = "references"
    APPROVED_BY = "approved_by"
    REJECTED_BY = "rejected_by"
    SUPERSEDES = "supersedes"
    BLOCKED_BY = "blocked_by"
    ASSIGNED_TO = "assigned_to"
```

### 3.3.7 MemoryView

MemoryView 是 Agent Process 当前可见对象集合。

```python
@dataclass
class MemoryView:
    view_id: MemoryViewID
    owner_pid: PID
    roots: list[ObjectHandle]
    filters: list[ObjectFilter]
    rights_policy: ViewRightsPolicy
    created_from: MemoryViewID | SnapshotID | None
    mode: ViewMode
```

```python
class ViewMode(Enum):
    READ_ONLY = "read_only"
    COPY_ON_WRITE = "copy_on_write"
    MUTABLE = "mutable"
    EPHEMERAL = "ephemeral"
```

### 3.3.8 Object Memory API

```python
class ObjectMemoryAPI:
    def create_object(
            self,
            pid: PID,
            type: ObjectType,
            payload: Any,
            metadata: ObjectMetadata | None = None,
            immutable: bool = True,
    ) -> ObjectHandle: ...

    def get_object(self, pid: PID, handle: ObjectHandle) -> AgentObject: ...

    def update_object(
            self,
            pid: PID,
            handle: ObjectHandle,
            patch: ObjectPatch,
    ) -> ObjectHandle: ...

    def link_objects(
            self,
            pid: PID,
            src: ObjectHandle,
            relation: RelationType,
            dst: ObjectHandle,
            metadata: dict | None = None,
    ) -> None: ...

    def query_objects(
            self,
            pid: PID,
            query: ObjectQuery,
    ) -> list[ObjectHandle]: ...

    def create_view(
            self,
            pid: PID,
            roots: list[ObjectHandle],
            mode: ViewMode,
            filters: list[ObjectFilter] | None = None,
    ) -> MemoryView: ...

    def fork_view(
            self,
            parent_pid: PID,
            child_pid: PID,
            parent_view: MemoryView,
            spec: MemoryViewSpec,
    ) -> MemoryView: ...

    def merge_view(
            self,
            parent_pid: PID,
            child_view: MemoryView,
            policy: MergePolicy,
    ) -> MergeResult: ...

    def snapshot_view(self, pid: PID, view: MemoryView) -> SnapshotID: ...

    def materialize_context(
            self,
            pid: PID,
            view: MemoryView,
            policy: ContextPolicy,
            budget_tokens: int,
    ) -> MaterializedContext: ...
```

### 3.3.9 Context Materialization

LLM 不能直接访问整个 Object Store。每次模型调用前，ContextMaterializer 将 MemoryView 转换为模型上下文。

```python
@dataclass
class MaterializedContext:
    text: str
    object_refs: list[OID]
    token_count: int
    omitted_objects: list[OID]
    policy_used: ContextPolicy
```

Materialization 策略：

- `evidence_first`：证据优先；
- `recency_first`：最近对象优先；
- `plan_first`：计划与当前状态优先；
- `error_debug`：错误日志与相关代码优先；
- `human_constraints_first`：人类约束和授权优先；
- `minimal`：只放入任务必要对象；
- `full_debug`：尽可能完整，用于调试。

---

## 3.4 Event System

### 3.4.1 设计目标

Event System 负责处理：

- 进程间通信；
- 人类中断；
- 工具返回；
- capability 变化；
- 定时器；
- checkpoint；
- 子进程退出；
- 外部环境变化。

### 3.4.2 Event

```python
@dataclass(frozen=True)
class Event:
    event_id: EventID
    type: EventType
    source: ActorRef
    target: ActorRef | None
    payload: dict
    priority: EventPriority
    created_at: Timestamp
    correlation_id: CorrelationID | None
    causality: list[EventID]
```

### 3.4.3 EventType

```python
class EventType(Enum):
    PROCESS_CREATED = "process_created"
    PROCESS_EXITED = "process_exited"
    PROCESS_FAILED = "process_failed"
    PROCESS_SIGNAL = "process_signal"

    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_CALL_RESULT = "tool_call_result"
    TOOL_CALL_FAILED = "tool_call_failed"

    HUMAN_QUERY = "human_query"
    HUMAN_RESPONSE = "human_response"
    HUMAN_INTERRUPT = "human_interrupt"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"

    CAPABILITY_GRANTED = "capability_granted"
    CAPABILITY_REVOKED = "capability_revoked"
    CAPABILITY_DENIED = "capability_denied"

    SKILL_LOADED = "skill_loaded"
    SKILL_UNLOADED = "skill_unloaded"

    CHECKPOINT_CREATED = "checkpoint_created"
    ROLLBACK_PERFORMED = "rollback_performed"

    MEMORY_OBJECT_CREATED = "memory_object_created"
    MEMORY_OBJECT_UPDATED = "memory_object_updated"
    MEMORY_VIEW_MERGED = "memory_view_merged"

    TIMER_EXPIRED = "timer_expired"
    POLICY_VIOLATION = "policy_violation"
    RESOURCE_EXHAUSTED = "resource_exhausted"
```

### 3.4.4 Event API

```python
class EventAPI:
    def send(
            self,
            source: ActorRef,
            target: ActorRef,
            type: EventType,
            payload: dict,
            priority: EventPriority = EventPriority.NORMAL,
    ) -> EventID: ...

    def recv(
            self,
            pid: PID,
            filter: EventFilter | None = None,
            timeout: Duration | None = None,
    ) -> Event | None: ...

    def poll(self, pid: PID, filter: EventFilter | None = None) -> list[Event]: ...

    def subscribe(self, pid: PID, filter: EventFilter) -> SubscriptionID: ...

    def interrupt(
            self,
            source: ActorRef,
            target_pid: PID,
            signal: ProcessSignal,
            payload: dict | None = None,
            priority: EventPriority = EventPriority.HIGH,
    ) -> EventID: ...

    def ack(self, pid: PID, event_id: EventID) -> None: ...
```

### 3.4.5 中断语义

中断分为四类：

| 类型            | 处理方式   | 示例            |
|---------------|--------|---------------|
| Immediate     | 立即抢占   | 停止删除文件、撤销网络调用 |
| SafePoint     | 到安全点处理 | 修改目标、切换策略     |
| Deferred      | 延迟生效   | 更新偏好、补充背景信息   |
| Informational | 不改变执行  | 查询状态、请求解释     |

```python
class InterruptClass(Enum):
    IMMEDIATE = "immediate"
    SAFE_POINT = "safe_point"
    DEFERRED = "deferred"
    INFORMATIONAL = "informational"
```

所有立即中断都必须进入 audit log，并触发 checkpoint 或状态 dump。

---

## 3.5 Capability System

## 3.5.1 设计目标

Capability System 是安全核心。

它控制：

- 对象访问；
- 文件系统访问；
- 网络访问；
- shell 执行；
- 人类访问；
- skill loading；
- tool calling；
- tool registration；
- fork/exec；
- memory write；
- external side effects。

### 3.5.2 Capability

```python
@dataclass(frozen=True)
class Capability:
    cap_id: CapabilityID
    subject: ActorRef
    resource: ResourceRef
    rights: set[Right]
    constraints: list[CapabilityConstraint]
    issued_by: ActorRef
    issued_at: Timestamp
    expires_at: Timestamp | None
    delegable: bool
    revocable: bool
```

### 3.5.3 Rights

```python
class Right(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    LIST = "list"
    NETWORK = "network"
    SHELL = "shell"
    HUMAN_QUERY = "human_query"
    HUMAN_INTERRUPT = "human_interrupt"
    LOAD_SKILL = "load_skill"
    REGISTER_TOOL = "register_tool"
    CALL_TOOL = "call_tool"
    SPAWN_PROCESS = "spawn_process"
    GRANT_CAPABILITY = "grant_capability"
    REVOKE_CAPABILITY = "revoke_capability"
    PERSIST_MEMORY = "persist_memory"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
```

### 3.5.4 Capability API

```python
class CapabilityAPI:
    def request(
            self,
            pid: PID,
            resource: ResourceRef,
            rights: set[Right],
            reason: str,
            duration: Duration | None = None,
    ) -> CapabilityDecision: ...

    def grant(
            self,
            issuer: ActorRef,
            subject: ActorRef,
            resource: ResourceRef,
            rights: set[Right],
            constraints: list[CapabilityConstraint] | None = None,
            duration: Duration | None = None,
    ) -> Capability: ...

    def revoke(self, issuer: ActorRef, cap_id: CapabilityID, reason: str) -> None: ...

    def check(
            self,
            subject: ActorRef,
            resource: ResourceRef,
            right: Right,
            context: dict | None = None,
    ) -> bool: ...

    def delegate(
            self,
            pid: PID,
            cap_id: CapabilityID,
            target: ActorRef,
            attenuation: CapabilityAttenuation,
    ) -> Capability: ...

    def attenuate(
            self,
            cap: Capability,
            attenuation: CapabilityAttenuation,
    ) -> Capability: ...
```

### 3.5.5 fork 时的 capability 继承

默认规则：

```text
子进程不自动继承父进程全部 capability。
fork 必须显式指定 capability inheritance policy。
所有 inherited capability 默认 attenuate。
高风险 capability 不可自动继承。
```

```python
class CapabilityInheritancePolicy(Enum):
    NONE = "none"
    READ_ONLY = "read_only"
    EXPLICIT = "explicit"
    ATTENUATED = "attenuated"
    FULL_TRUSTED = "full_trusted"
```

MVP 默认使用 `EXPLICIT` 或 `ATTENUATED`。

### 3.5.6 高风险 capability

以下 capability 默认需要人类授权或管理员策略授权：

- 任意网络访问；
- shell 写操作；
- 删除文件；
- 修改 git history；
- 发送邮件；
- 注册持久工具；
- 加载未签名 skill；
- 写入长期记忆；
- 访问凭据；
- fork 大量子进程；
- exec 到未信任镜像；
- 访问敏感对象。

---

## 3.6 Skills / Tools Layer

## 3.6.1 分层定位

Skills / Tools Layer 是 Agent personality 与 Agent LibOS 之间的 LLM-facing capability layer。

它不应该被理解为 Host Runtime 的一部分，也不应该直接等同于 libOS kernel ABI。它的职责是把底层原语包装成模型可调用的工具、技能和组合动作。

这一层包括两类能力：

- **Skill**：改变 Agent 如何理解、计划、判断、压缩上下文和使用工具；
- **Tool**：暴露一个可调用动作，可能访问外部环境或触发 libOS 原语组合。

底层 libOS 原语应该保持小而稳定；Skills / Tools Layer 可以快速演化、按领域扩展、按任务加载。

## 3.6.2 Skill 定义

Skill 是动态链接到 Agent Process 的能力模块。

Skill 不应直接等同于 Tool。

区别：

| 概念         | 类比                               | 作用                    |
|------------|----------------------------------|-----------------------|
| Tool       | function call / external service | 访问外部世界，可能有副作用         |
| Skill      | dynamic library                  | 增强 Agent 内部能力、策略和领域知识 |
| AgentImage | executable image                 | 定义进程执行身份和默认行为         |
| Subagent   | child process                    | 隔离执行子任务               |

### 3.6.3 SkillObject

```python
@dataclass(frozen=True)
class SkillObject:
    skill_id: SkillID
    name: str
    version: str
    description: str
    instructions: str
    examples: list[SkillExample]
    resources: list[ObjectHandle]
    scripts: list[ScriptRef]
    required_capabilities: list[CapabilityRequirement]
    compatible_images: list[AgentImageID]
    metadata: dict
    signature: Signature | None
```

### 3.6.4 Skill API

```python
class SkillAPI:
    def discover(
            self,
            pid: PID,
            query: SkillQuery,
    ) -> list[SkillRef]: ...

    def load(
            self,
            pid: PID,
            skill: SkillRef,
            mode: SkillLoadMode = SkillLoadMode.LAZY,
    ) -> SkillHandle: ...

    def unload(self, pid: PID, handle: SkillHandle) -> None: ...

    def resolve(
            self,
            pid: PID,
            symbol: str,
    ) -> SkillSymbol | None: ...

    def verify(self, skill: SkillRef) -> VerificationResult: ...

    def pin_version(self, pid: PID, skill: SkillRef, version: str) -> None: ...
```

### 3.6.5 Skill 加载语义

Skill load 必须经过：

- schema/version compatibility check；
- signature/provenance check；
- capability requirement check；
- prompt injection scan；
- resource access check；
- audit log。

默认使用 lazy loading：

```text
Skill 元数据先进入上下文；
完整 instructions/examples/scripts 仅在需要时 materialize。
```

---

## 3.7 Tool Broker 与 JIT Tool

## 3.7.1 设计目标

Tool Broker 负责管理所有工具调用和工具注册。

Agent 不应直接获得任意代码执行权。它可以提出工具候选，但必须由 Tool Broker 进行验证、构建、测试、签名和授权。

需要注意：Tool Broker 属于 libOS/runtime 的安全边界；而具体暴露给 LLM 的 tool/action 属于 Skills / Tools Layer。二者关系为：

```text
ToolBroker: build / verify / sandbox / register / revoke
SkillsToolsLayer: describe / wrap / compose / expose to LLM
```

因此，工具注册完成后，还需要生成 LLM-facing wrapper，包括名称、说明、schema、示例、权限提示、失败模式和使用策略。

### 3.7.2 ToolSpec

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    side_effects: list[SideEffect]
    required_capabilities: list[CapabilityRequirement]
    timeout: Duration
    resource_limits: ResourceLimits
    deterministic: bool
    idempotent: bool
```

### 3.7.3 ToolCandidateObject

```python
@dataclass(frozen=True)
class ToolCandidateObject:
    candidate_id: ToolCandidateID
    spec: ToolSpec
    source_code: str
    tests: list[ToolTest]
    build_config: BuildConfig
    requested_capabilities: list[CapabilityRequirement]
    created_by: PID
    provenance: Provenance
```

### 3.7.4 ToolHandle

```python
@dataclass(frozen=True)
class ToolHandle:
    tool_id: ToolID
    version: str
    rights: set[Right]
    capability_id: CapabilityID
    sandbox_profile: SandboxProfile
    expires_at: Timestamp | None
```

### 3.7.5 Tool API

```python
class ToolAPI:
    def call(
            self,
            pid: PID,
            tool: ToolHandle,
            args: dict,
            timeout: Duration | None = None,
    ) -> ToolCallID: ...

    def get_result(self, pid: PID, call_id: ToolCallID) -> ToolResultObject: ...

    def propose(
            self,
            pid: PID,
            spec: ToolSpec,
            source_code: str,
            tests: list[ToolTest],
            requested_capabilities: list[CapabilityRequirement],
    ) -> ToolCandidateID: ...

    def validate(self, candidate: ToolCandidateID) -> ValidationResult: ...

    def register(
            self,
            approver: ActorRef,
            candidate: ToolCandidateID,
            scope: ToolScope,
    ) -> ToolHandle: ...

    def revoke(self, issuer: ActorRef, tool_id: ToolID, reason: str) -> None: ...
```

### 3.7.6 JIT Tool 注册流水线

```text
1. Agent 创建 ToolCandidateObject
2. ToolBroker 进行 schema 检查
3. Sandbox 构建环境
4. 执行静态分析
5. 执行单元测试
6. 执行资源限制测试
7. 分析 requested capabilities
8. PolicyEngine 判断是否需要 human approval
9. HumanObject 批准/拒绝/修改权限
10. ToolRegistry 签名并注册
11. Agent 获得受限 ToolHandle
12. AuditManager 记录完整链路
```

### 3.7.7 Tool Scope

```python
class ToolScope(Enum):
    EPHEMERAL_PROCESS = "ephemeral_process"
    TASK_LOCAL = "task_local"
    USER_LOCAL = "user_local"
    PROJECT_LOCAL = "project_local"
    GLOBAL_SIGNED = "global_signed"
```

MVP 只允许：

- `EPHEMERAL_PROCESS`；
- `TASK_LOCAL`。

`PROJECT_LOCAL` 和 `GLOBAL_SIGNED` 需要更严格治理。

---

## 3.8 HumanObject

## 3.8.1 定义

HumanObject 是外部对象、权限持有者和中断源。

它不是普通工具。

HumanObject 支持：

- 被 Agent query；
- 回复 approval/rejection/edit；
- 主动 interrupt Agent；
- grant/revoke capability；
- inspect state；
- override goal；
- request explanation。

### 3.8.2 HumanObject 数据模型

```python
@dataclass
class HumanObject:
    human_id: HumanID
    display_name: str
    roles: list[HumanRole]
    authority: AuthorityProfile
    contact_channels: list[ContactChannel]
    availability_policy: AvailabilityPolicy
    interruption_cost: InterruptionCostModel
    preferences_ref: ObjectHandle | None
```

### 3.8.3 HumanRequest

```python
@dataclass(frozen=True)
class HumanRequest:
    request_id: HumanRequestID
    pid: PID
    type: HumanRequestType
    question: str
    context_objects: list[ObjectHandle]
    options: list[HumanOption] | None
    expected_schema: dict | None
    default_action: HumanDefaultAction | None
    deadline: Timestamp | None
    blocking: bool
    risk_level: RiskLevel
    created_at: Timestamp
```

```python
class HumanRequestType(Enum):
    CLARIFICATION = "clarification"
    APPROVAL = "approval"
    PREFERENCE = "preference"
    AUTHORIZATION = "authorization"
    CONSTRAINT_UPDATE = "constraint_update"
    STATUS_REVIEW = "status_review"
    EXCEPTION_HANDLING = "exception_handling"
```

### 3.8.4 Human API

```python
class HumanAPI:
    def query(
            self,
            pid: PID,
            human: HumanID,
            request: HumanRequest,
    ) -> HumanRequestID: ...

    def receive_response(
            self,
            request_id: HumanRequestID,
            response: HumanResponse,
    ) -> None: ...

    def interrupt(
            self,
            human: HumanID,
            target_pid: PID,
            signal: ProcessSignal,
            payload: dict | None = None,
    ) -> EventID: ...

    def inspect(
            self,
            human: HumanID,
            pid: PID,
            scope: InspectScope,
    ) -> InspectionResult: ...

    def approve(
            self,
            human: HumanID,
            request_id: HumanRequestID,
            decision: ApprovalDecision,
    ) -> None: ...

    def grant_capability(
            self,
            human: HumanID,
            pid: PID,
            resource: ResourceRef,
            rights: set[Right],
            constraints: list[CapabilityConstraint],
    ) -> Capability: ...

    def revoke_capability(
            self,
            human: HumanID,
            cap_id: CapabilityID,
            reason: str,
    ) -> None: ...
```

### 3.8.5 Human Interrupt 类型

```python
class HumanSignal(Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    CHANGE_GOAL = "change_goal"
    ADD_CONSTRAINT = "add_constraint"
    REVOKE_CAPABILITY = "revoke_capability"
    REQUEST_STATUS = "request_status"
    REQUEST_EXPLANATION = "request_explanation"
    ROLLBACK = "rollback"
    APPROVE_PENDING = "approve_pending"
    REJECT_PENDING = "reject_pending"
```

---

## 3.9 Checkpoint 与 Rollback

## 3.9.1 Checkpoint 内容

Agent checkpoint 不保存字节级内存，而是保存对象图 root 和运行状态引用。

```python
@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: CheckpointID
    pid: PID
    image: AgentImageRef
    process_state: ObjectHandle
    goal: ObjectHandle
    memory_view: MemoryView
    capability_snapshot: CapabilitySnapshot
    loaded_skills: list[SkillHandle]
    tool_table: list[ToolHandle]
    event_cursor: EventCursor
    pending_requests: list[HumanRequestID | ToolCallID]
    created_at: Timestamp
    reason: str
```

### 3.9.2 Checkpoint API

```python
class CheckpointAPI:
    def create(self, pid: PID, reason: str) -> CheckpointID: ...

    def restore(self, pid: PID, checkpoint: CheckpointID) -> None: ...

    def rollback(
            self,
            pid: PID,
            checkpoint: CheckpointID,
            rollback_policy: RollbackPolicy,
    ) -> RollbackResult: ...

    def diff(self, a: CheckpointID, b: CheckpointID) -> CheckpointDiff: ...

    def list(self, pid: PID) -> list[CheckpointSummary]: ...
```

### 3.9.3 Rollback 限制

不是所有外部副作用都可回滚。

必须区分：

- object memory rollback；
- process state rollback；
- tool table rollback；
- filesystem diff rollback；
- external side-effect compensation。

例如，发送邮件不可真正撤销，只能记录 compensation action。

---

## 3.10 Primitive Managers 与 Resource Provider Substrate

## 3.10.1 Primitive 类型

文件系统不是根抽象，而是 libOS primitive 之一。Primitive 负责 libOS 语义，Resource Provider Substrate 负责真实宿主资源访问。

```text
Primitive Managers:
  FilesystemAdapter
  ShellAdapter
  ClockPrimitive
  HumanObjectManager
  ImageRegistryPrimitive
  ProcessManager
  ObjectMemoryManager
  Future Browser/Git/Database/IDE/Calendar/Mail/Search primitives
```

Primitive 不应直接等同于底层 Host OS 调用。Primitive 负责 libOS 语义，例如 capability 检查、人类授权、审计和事件；
真正的资源访问应委托给可替换的 Resource Provider Substrate：

```python
class ResourceProviderSubstrate:
    filesystem: FilesystemProvider
    clock: ClockProvider
    shell: ShellProvider
    human: HumanProvider
```

MVP 默认实现可以是本地 host-backed provider；后续可以替换为容器、远程执行环境、WASM sandbox、云对象存储或其它
Resource Provider Substrate，而不改变 Agent 可见工具、process capability 模型和 audit/event 语义。
HumanObjectManager 仍负责请求队列、审批状态机、process wakeup、capability/audit/event 语义；终端或 UI 的实际读写由
HumanProvider 承担。

### 3.10.2 ExternalObjectRef

```python
@dataclass(frozen=True)
class ExternalObjectRef:
    adapter: str
    external_id: str
    type: str
    metadata: dict
```

外部实体需要进入 Object Memory 时，通过 ExternalRefObject 保存可审计引用和快照，而不是绕过 primitive/provider 边界：

```python
@dataclass(frozen=True)
class ExternalRefPayload:
    ref: ExternalObjectRef
    snapshot: ObjectHandle | None
    last_observed_at: Timestamp
    consistency: ConsistencyModel
```

---

## 4. 调度与执行模型

## 4.1 Scheduler 目标

Scheduler 负责决定哪个 Agent Process 可以执行、执行多久、何时暂停、何时处理事件。

调度依据：

- process status；
- priority；
- deadline；
- resource budget；
- waiting events；
- human availability；
- risk level；
- pending approvals；
- child process dependencies。

## 4.2 Execution Quantum

Agent Process 每次执行一个 quantum。若该进程消息队列中存在 unread interrupt message，runtime 必须在实际 tool 调用前通知进程并抢占非消息读取类 tool；若存在 unread normal message，runtime 应在当前 tool 调用结束后通知进程。消息本体通过显式 `read_process_messages` / `process.read_messages` 查看，读取默认 ack，避免把队列语义退化成不可控 prompt 文本。若进程需要等待 IPC，可调用 `receive_process_messages` / `process.receive_messages` 并按 kind、sender、channel、correlation_id、reply_to 或 message_ids 做选择性匹配；无匹配消息时进程进入 `WAITING_EVENT`，匹配消息到达后由消息队列 manager 唤醒并恢复原 action/syscall。人类主动输入也应由 HumanObject primitive 作为 `human:<name>` sender 投递到同一进程 mailbox，包括普通消息和 interrupt；不得直接拼接进 prompt 或绕过消息队列。

一个 quantum 可能包括：

- materialize context；
- call LLM；
- parse action；
- validate action；
- dispatch one action；
- update state；
- checkpoint if needed。

MVP 中建议每个 quantum 最多产生一个外部副作用 action。

## 4.3 Safe Point

Safe Point 是可处理中断、创建 checkpoint、合并子任务和接受人类修改的位置。

Safe Point 发生在：

- LLM action 之前；
- tool call 之前；
- tool result 之后；
- memory merge 前；
- JIT tool register 前；
- high-risk side effect 前；
- child process join 后；
- human response 后。

## 4.4 Blocking vs Non-blocking Human Requests

Human request 分两类：

- blocking：没有回复不能继续；
- non-blocking：Agent 可继续做低风险替代任务。

默认规则：

```text
审批类请求 blocking。
偏好类请求可 non-blocking。
状态询问不阻塞。
高风险授权必须 blocking。
```

---

## 5. 安全模型

## 5.1 威胁模型

需要防范：

1. Agent 误操作；
2. prompt injection；
3. tool output injection；
4. skill poisoning；
5. JIT tool 恶意或错误代码；
6. capability bypass；
7. fork bomb；
8. resource exhaustion；
9. credential leakage；
10. unsafe external side effects；
11. human approval spoofing；
12. audit log tampering。

## 5.2 安全边界

### 5.2.1 Agent Process 边界

Agent Process 不能直接访问宿主资源。所有访问必须经由 capability。

### 5.2.2 Tool Sandbox 边界

JIT tool 和高风险工具必须运行在 sandbox 中。

Sandbox 需要限制：

- filesystem；
- network；
- environment variables；
- CPU；
- memory；
- wall time；
- subprocess；
- credentials。

### 5.2.3 Skill 边界

Skill 是能力模块，不应默认获得外部副作用权限。

Skill 中包含的脚本也必须通过 ToolBroker 或 Sandbox 执行。

### 5.2.4 Human Authority 边界

不同 human role 拥有不同授权能力。

```python
class HumanRole(Enum):
    OWNER = "owner"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    OPERATOR = "operator"
    OBSERVER = "observer"
```

例如：

- OWNER 可以 grant/revoke 高风险 capability；
- REVIEWER 可以 approve code patch；
- OBSERVER 只能 inspect status。

## 5.3 Policy Engine

PolicyEngine 接收 action proposal，返回 decision。

```python
class PolicyDecision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HUMAN_APPROVAL = "require_human_approval"
    REQUIRE_SANDBOX = "require_sandbox"
    REQUIRE_CHECKPOINT = "require_checkpoint"
    REQUIRE_CAPABILITY_ATTENUATION = "require_capability_attenuation"
```

```python
class PolicyEngine:
    def evaluate_action(
            self,
            pid: PID,
            action: ActionProposal,
            context: PolicyContext,
    ) -> PolicyDecisionBundle: ...
```

## 5.4 默认安全规则

1. 默认无网络；
2. 默认无 shell；
3. 默认不能写长期记忆；
4. 默认不能注册持久工具；
5. 默认不能加载未签名 skill；
6. fork 默认权限收缩；
7. exec 不提升权限；
8. JIT tool 默认 ephemeral；
9. 高风险操作前自动 checkpoint；
10. 所有人类授权必须进入 audit log。

---

## 6. 审计与可观测性

## 6.1 Audit Record

```python
@dataclass(frozen=True)
class AuditRecord:
    record_id: AuditID
    timestamp: Timestamp
    actor: ActorRef
    action: str
    target: ResourceRef | None
    input_refs: list[OID]
    output_refs: list[OID]
    capability_refs: list[CapabilityID]
    decision: str | None
    policy_decision: PolicyDecisionBundle | None
    correlation_id: CorrelationID | None
    parent_record_id: AuditID | None
```

## 6.2 必须审计的操作

- process fork/exec/exit/kill；
- capability grant/revoke/request/deny；
- skill load/unload；
- tool call；
- tool propose/register/revoke；
- human query/response/approval/interrupt；
- external side effect；
- memory persistent write；
- checkpoint/rollback；
- policy violation；
- sandbox failure。

## 6.3 Trace 查询

支持回答：

- 某个修改为什么发生？
- 哪个人批准了这个操作？
- 某个子 Agent 继承了哪些 capability？
- 某个 JIT tool 是谁创建的，测试结果如何？
- 某个对象被哪些进程读取过？
- rollback 会影响哪些对象？

---

## 7. 最小可行版本 MVP

## 7.1 MVP 目标

第一版不要实现完整 Agent OS。目标是验证核心抽象：

1. Agent Process 可运行；
2. Object Memory 可创建、查询、materialize；
3. capability 可以限制工具调用；
4. human interrupt/approval 可暂停恢复；
5. JIT tool 可在 sandbox 中生成、测试、注册为 ephemeral tool；
6. fork worker 子进程可执行封闭任务；
7. audit log 可追踪关键动作。

## 7.2 MVP 模块清单

### 必须实现

- ProcessManager；
- EventBus；
- ObjectMemoryManager；
- ContextMaterializer；
- CapabilityManager；
- ToolBroker；
- HumanObjectManager；
- AuditManager；
- SimpleScheduler；
- SQLite/Postgres-backed store；
- Docker/Firecracker-like sandbox abstraction，初版可用 Docker；
- CLI/Web debug console。

### 暂不实现

- 完整多租户；
- 分布式调度；
- 复杂 actor runtime；
- 全局 skill marketplace；
- 持久化全局 JIT tool registry；
- 自动长期记忆优化；
- 复杂 rollback compensation；
- formal verification。

## 7.3 MVP API 原语

```python
# process
fork(goal, memory_view, capabilities) -> PID
exec(pid, image) -> None
signal(pid, signal) -> None
wait(pid) -> ProcessResult

# memory
create_object(type, payload) -> ObjectHandle
get_object(handle) -> AgentObject
link_objects(src, relation, dst) -> None
query_objects(query) -> list[ObjectHandle]
create_view(roots) -> MemoryView
materialize_context(view, budget) -> MaterializedContext

# capability
request(resource, rights, reason) -> Decision
grant(subject, resource, rights) -> Capability
revoke(capability) -> None
check(subject, resource, right) -> bool

# human
query(human, request) -> HumanRequestID
interrupt(pid, signal) -> EventID
approve(request_id, decision) -> None

# tool
call(tool_handle, args) -> ToolCallID
propose_tool(spec, code, tests) -> ToolCandidateID
validate_tool(candidate) -> ValidationResult
register_tool(candidate) -> ToolHandle

# checkpoint
checkpoint(pid, reason) -> CheckpointID
rollback(pid, checkpoint) -> RollbackResult
```

## 7.4 MVP 参考任务

选择一个 coding-agent demo：

```text
目标：修复一个小型 Python/Rust 项目的 failing tests。
```

流程：

1. Root Agent 创建 task object；
2. 读取 repo summary、test log；
3. fork worker 分析错误日志；
4. worker 返回 ErrorTrace object；
5. Root Agent 生成 patch；
6. 调用 test tool；
7. 如需专门解析日志，propose JIT parser tool；
8. ToolBroker sandbox 测试 parser；
9. 注册 ephemeral parser；
10. Agent 使用 parser 辅助调试；
11. 高风险修改前请求 human approval；
12. human approve；
13. 应用 patch；
14. 运行测试；
15. 生成 audit trace 和 final report。

这个 demo 可以覆盖 process、memory、tool、human、capability、audit 的核心链路。

---

## 8. 目录结构建议

```text
agent_libos/
  skills_tools/
    skill_registry.py
    tool_registry.py
    tool_bundle.py
    action_schema.py
    wrappers.py
    macros.py
    package_loader.py

  runtime/
    process_manager.py
    scheduler.py
    event_bus.py
    audit_manager.py
    checkpoint_manager.py

  memory/
    object_store.py
    object_graph.py
    memory_view.py
    materializer.py
    schemas.py

  capability/
    manager.py
    policy.py
    rights.py
    constraints.py

  skills/
    linker.py
    registry.py
    verifier.py
    schema.py

  tools/
    broker.py
    registry.py
    sandbox.py
    validator.py
    schemas.py

  human/
    manager.py
    requests.py
    interrupts.py
    ui_adapter.py

  primitives/
    filesystem.py
    shell.py
    git.py
    browser.py
    database.py

  images/
    base_agent.py
    coding_agent.py
    toolmaker_agent.py
    review_agent.py

  llm/
    client.py
    action_parser.py
    context_protocol.py

  storage/
    postgres.py
    sqlite.py
    blob_store.py

  api/
    python_sdk.py
    server.py
    cli.py

  tests/
    unit/
    integration/
    sandbox/
    security/
```

---

## 9. 数据库与存储建议

## 9.1 存储层

MVP 可使用：

- Postgres：metadata、objects、links、events、capabilities、audit；
- S3/MinIO/local blob store：大 payload、tool artifacts、logs；
- Redis/NATS：事件队列，MVP 可先用 Postgres queue；
- Vector DB：对象 embedding，MVP 可先用 pgvector。

## 9.2 表结构草案

```sql
CREATE TABLE objects
(
    oid            TEXT PRIMARY KEY,
    type           TEXT      NOT NULL,
    schema_version TEXT      NOT NULL,
    payload_ref    TEXT,
    payload_json   JSONB,
    metadata       JSONB     NOT NULL,
    provenance     JSONB     NOT NULL,
    version        INTEGER   NOT NULL,
    immutable      BOOLEAN   NOT NULL,
    created_by     TEXT      NOT NULL,
    created_at     TIMESTAMP NOT NULL,
    updated_at     TIMESTAMP NOT NULL
);

CREATE TABLE object_links
(
    id         TEXT PRIMARY KEY,
    src_oid    TEXT      NOT NULL,
    relation   TEXT      NOT NULL,
    dst_oid    TEXT      NOT NULL,
    metadata   JSONB,
    created_by TEXT      NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE processes
(
    pid            TEXT PRIMARY KEY,
    parent_pid     TEXT,
    image_id       TEXT      NOT NULL,
    status         TEXT      NOT NULL,
    goal_oid       TEXT,
    memory_view_id TEXT,
    capabilities   JSONB,
    loaded_skills  JSONB,
    tool_table     JSONB,
    event_cursor   TEXT,
    created_at     TIMESTAMP NOT NULL,
    updated_at     TIMESTAMP NOT NULL
);

CREATE TABLE events
(
    event_id       TEXT PRIMARY KEY,
    type           TEXT      NOT NULL,
    source         TEXT      NOT NULL,
    target         TEXT,
    payload        JSONB     NOT NULL,
    priority       INTEGER   NOT NULL,
    created_at     TIMESTAMP NOT NULL,
    correlation_id TEXT,
    causality      JSONB
);

CREATE TABLE capabilities
(
    cap_id      TEXT PRIMARY KEY,
    subject     TEXT      NOT NULL,
    resource    TEXT      NOT NULL,
    rights      JSONB     NOT NULL,
    constraints JSONB,
    issued_by   TEXT      NOT NULL,
    issued_at   TIMESTAMP NOT NULL,
    expires_at  TIMESTAMP,
    delegable   BOOLEAN   NOT NULL,
    revocable   BOOLEAN   NOT NULL,
    revoked     BOOLEAN   NOT NULL DEFAULT FALSE
);

CREATE TABLE audit_records
(
    record_id        TEXT PRIMARY KEY,
    timestamp        TIMESTAMP NOT NULL,
    actor            TEXT      NOT NULL,
    action           TEXT      NOT NULL,
    target           TEXT,
    input_refs       JSONB,
    output_refs      JSONB,
    capability_refs  JSONB,
    decision         JSONB,
    correlation_id   TEXT,
    parent_record_id TEXT
);

CREATE TABLE llm_calls
(
    call_id          TEXT PRIMARY KEY,
    pid              TEXT,
    image_id         TEXT,
    purpose          TEXT      NOT NULL,
    status           TEXT      NOT NULL,
    api              TEXT,
    model            TEXT,
    request_id       TEXT,
    response_id      TEXT,
    messages         JSONB     NOT NULL,
    tools            JSONB     NOT NULL,
    request_options  JSONB     NOT NULL,
    response_content TEXT      NOT NULL,
    tool_calls       JSONB     NOT NULL,
    reasoning        JSONB,
    usage            JSONB     NOT NULL,
    raw_response     JSONB,
    error            TEXT,
    created_at       TIMESTAMP NOT NULL,
    completed_at     TIMESTAMP
);
```

真实 LLM 调用是稀缺且可计费资源。除 audit 摘要外，runtime 必须把每次请求的完整输入、可见 tool schema、模型输出、tool call、usage/token 统计、provider 暴露的 reasoning 字段、raw response 和错误信息持久化到 `llm_calls` 一类表中，供复盘、成本核算和调试使用。

---

## 10. 开发阶段规划

## Phase 0：概念验证

目标：跑通单 Agent + Object Memory + Tool Call。

交付：

- Python SDK；
- ObjectStore；
- ProcessManager 单进程版本；
- ToolBroker 调用静态工具；
- AuditLog；
- 简单 CLI。

验收：

- 能创建 task object；
- 能 materialize context；
- 能调用 read/test 工具；
- 所有动作有 audit record。

## Phase 1：Human Interrupt + Capability

目标：加入人类授权和中断。

交付：

- CapabilityManager；
- HumanRequest；
- interrupt/resume；
- approval flow；
- debug console。

验收：

- 高风险工具调用会被拦截；
- 人类 approve 后恢复执行；
- 人类 pause/cancel 可以中断进程；
- capability grant/revoke 可追踪。

## Phase 2：fork worker + MemoryView

目标：实现子 Agent 和受限内存视图。

交付：

- fork restricted/worker；
- wait/join；
- MemoryView fork；
- child result merge；
- resource budget。

验收：

- Root Agent 可 fork worker 分析日志；
- worker 无法访问未授权对象；
- worker 输出可合并回 root memory；
- fork 行为有 audit trace。

## Phase 3：Skill Linker

目标：实现动态 skill 加载。

交付：

- Skill schema；
- Skill registry；
- lazy loading；
- compatibility check；
- skill audit。

验收：

- Agent 可 discover/load/unload skill；
- skill 不自动获得外部副作用权限；
- materializer 能按需展开 skill 内容。

## Phase 4：JIT Tool

目标：实现受控工具生成。

交付：

- ToolCandidateObject；
- sandbox build/test；
- static checks；
- ephemeral registration；
- ToolHandle；
- human approval integration。

验收：

- Agent 可提出日志解析工具；
- 工具在 sandbox 中测试；
- 通过后以 ephemeral tool 注册；
- Agent 可调用新工具；
- 注册过程完整审计。

## Phase 5：exec + checkpoint/rollback

目标：实现进程镜像切换和恢复。

交付：

- AgentImage registry；
- exec check；
- checkpoint create/restore；
- rollback object view；
- diff report。

验收：

- Agent 可从 base-agent exec 到 coding-agent；
- exec 不提升权限；
- checkpoint 可恢复状态；
- rollback 可撤销 object memory 修改。

---

## 11. 测试计划

## 11.1 单元测试

覆盖：

- ObjectStore CRUD；
- ObjectHandle 权限检查；
- ObjectLink 查询；
- MemoryView fork/merge；
- Capability grant/revoke/check；
- Event send/recv/interrupt；
- Process fork/exec/wait；
- Skill load verification；
- Tool candidate validation；
- Audit record 写入。

## 11.2 集成测试

场景：

1. 单 Agent 完成简单任务；
2. 高风险工具调用触发 human approval；
3. human interrupt pause/resume；
4. fork worker 分析输入；
5. JIT tool 生成并调用；
6. checkpoint 后 rollback；
7. exec 后继续执行。

## 11.3 安全测试

必须测试：

- 子进程越权访问对象；
- 子进程继承过多 capability；
- JIT tool 请求网络但未授权；
- JIT tool 访问凭据；
- skill 中包含 prompt injection；
- tool output injection；
- fork bomb；
- audit log tampering；
- revoked capability 继续使用；
- human approval spoofing。

## 11.4 性能测试

指标：

- object query latency；
- materialization latency；
- event dispatch latency；
- tool sandbox startup time；
- fork worker overhead；
- audit write throughput；
- token budget utilization；
- memory view merge cost。

---

## 12. 评估指标

## 12.1 任务指标

- task success rate；
- wall-clock time；
- number of tool calls；
- number of human interruptions；
- number of failed actions；
- rollback count；
- JIT tool reuse rate；
- child process usefulness。

## 12.2 安全指标

- unauthorized access attempts blocked；
- unnecessary capability grants；
- missed approval requirements；
- dangerous action prevention rate；
- audit completeness；
- rollback effectiveness。

## 12.3 人类交互指标

- unnecessary human query rate；
- average human response burden；
- approval latency；
- interruption recovery latency；
- human override success rate；
- status explainability score。

## 12.4 内存指标

- context materialization precision；
- object retrieval relevance；
- stale object usage rate；
- duplicate object rate；
- provenance completeness；
- merge conflict rate。

---

## 13. Python SDK 草案

```python
from agent_libos import Runtime, AgentImage, Rights

runtime = Runtime.open("local")

root = runtime.process.spawn(
    image="coding-agent:v0",
    goal={"text": "Fix failing tests in this repository"},
    capabilities=[
        runtime.capability.project_read("repo"),
        runtime.capability.tool_call("pytest", rights={Rights.EXECUTE}),
    ],
)

log_obj = runtime.memory.create_object(
    pid=root,
    type="error_trace",
    payload={"log": "..."},
)

worker = runtime.process.fork(
    parent=root,
    goal={"text": "Analyze the test failure log"},
    memory_view=runtime.memory.view([log_obj], mode="read_only"),
    capabilities=[],
    mode="worker",
)

result = runtime.process.wait(root, worker)

candidate = runtime.tools.propose(
    pid=root,
    spec={
        "name": "parse_pytest_log",
        "description": "Parse pytest failure logs into structured failures",
        "input_schema": {"type": "object", "properties": {"log": {"type": "string"}}},
        "output_schema": {"type": "array"},
    },
    source_code="...",
    tests=[...],
    requested_capabilities=[],
)

validation = runtime.tools.validate(candidate)
if validation.ok:
    tool = runtime.tools.register(
        approver="policy:local",
        candidate=candidate,
        scope="ephemeral_process",
    )

call = runtime.tools.call(root, tool, {"log": "..."})
parsed = runtime.tools.get_result(root, call)

runtime.human.query(
    pid=root,
    human="owner",
    request={
        "type": "approval",
        "question": "Apply this patch to the repository?",
        "context_objects": [parsed],
        "blocking": True,
        "risk_level": "medium",
    },
)
```

---

## 14. 需要尽早确定的设计决策

### 14.1 Agent Process 是强 actor 还是 workflow wrapper？

两种路线：

1. Actor-first：AgentProcess 是消息驱动 actor；
2. Workflow-first：AgentProcess 是持久 workflow 的包装。

建议 MVP 采用 workflow-first，内部保留 actor-like API。这样更容易实现 checkpoint、wait、human approval。

### 14.2 Object payload 存 JSON 还是 blob？

建议：

- 小对象：JSONB；
- 大对象：blob store；
- 所有对象都保留 metadata、summary、token_estimate。

### 14.3 Context materialization 是否可插拔？

必须可插拔。不同 AgentImage 应有不同 ContextPolicy。

### 14.4 JIT tool 支持哪些语言？

MVP 只支持 Python。后续支持 Rust/JS/WASM。

### 14.5 Sandbox 选型

MVP 可用 Docker，但接口要抽象为：

```python
class SandboxBackend:
    def build(...): ...

    def run(...): ...

    def inspect(...): ...

    def destroy(...): ...
```

后续可替换为 Firecracker、gVisor、WASM sandbox。

---

## 15. 非目标

第一阶段不追求：

- 完全自主的无限期 Agent；
- 无限制自我修改；
- 全自动全局工具市场；
- 完整 POSIX 兼容；
- 字节级内存模型；
- 文件系统作为根命名空间；
- 无人监管的高风险外部副作用；
- 多用户企业权限体系；
- 大规模分布式 agent cluster。

---

## 16. 总结

本框架的核心不是再做一个 workflow engine，也不是再做一个 tool-calling agent 框架，而是实现一个 Agent-native libOS 以及其上的
LLM-facing Skills / Tools Layer：

```text
Agent Process
  + Skills / Tools Layer
  + Object Memory
  + Event System
  + Capability Security
  + HumanObject Interrupts
  + Skill Dynamic Linking
  + JIT Tool Broker
  + Checkpoint/Rollback
  + Audit Trace
```

其中，libOS 提供稳定、可审计、受 capability 管控的底层原语；Skills / Tools Layer 则把这些原语包装成 LLM 能够可靠使用的
actions、skills、tool bundles 和 workflow macros。

最有辨识度的原语是：

```text
fork
exec
signal
checkpoint
rollback
fork_view
merge_view
materialize_context
dlopen_skill
propose_tool
register_tool
human_interrupt
capability_grant
capability_revoke
```

最重要的工程原则是：

> Agent 可以自主扩展执行能力，但所有能力扩展必须经过 capability-safe runtime control。

最重要的内存原则是：

> Agent memory is not byte-addressed memory and not a filesystem namespace; it is a typed, capability-protected,
> versioned object graph from which execution contexts are materialized.

如果团队按照本文档推进，第一阶段应优先做出一个可运行的 coding-agent demo，用最小系统验证：process、object
memory、capability、human interrupt、JIT tool 和 audit trace 是否能自然协作。
