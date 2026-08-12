import { McpServer, ResourceTemplate } from "@modelcontextprotocol/server";
import { StdioServerTransport, serveStdio } from "@modelcontextprotocol/server/stdio";
import fs from "node:fs";
import path from "node:path";
import { z } from "zod";


const RESOURCE_URI = "fixture://document/current";
const fixtureArgs = process.argv.slice(2);
if (fixtureArgs.length !== 0 &&
    (fixtureArgs.length !== 2 || fixtureArgs[0] !== "--task-state-file")) {
  throw new Error("invalid TypeScript MCP fixture arguments");
}
const taskStateFile = fixtureArgs.length === 2 ? path.resolve(fixtureArgs[1]) : null;
const taskNotificationSecret = process.env.MCP_FIXTURE_TASK_SECRET ?? "";
let taskExtensionHandler = null;

class TasksExtensionTransport extends StdioServerTransport {
  taskSubscriptionIds = new Set();

  async start() {
    const sdkMessageHandler = this.onmessage;
    this.onmessage = (message) => {
      const taskIds = message?.params?.notifications?.taskIds;
      if (message?.method === "subscriptions/listen" && Array.isArray(taskIds)) {
        if (message.id === undefined || taskIds.length === 0 ||
            taskIds.some((taskId) => typeof taskId !== "string") ||
            new Set(taskIds).size !== taskIds.length) {
          void this.send({
            jsonrpc: "2.0",
            id: message?.id ?? null,
            error: { code: -32602, message: "Invalid Task subscription" },
          });
          return;
        }
        this.taskSubscriptionIds.add(message.id);
        const metadata = {
          "io.modelcontextprotocol/subscriptionId": message.id,
        };
        void this.send({
          jsonrpc: "2.0",
          method: "notifications/subscriptions/acknowledged",
          params: {
            notifications: { taskIds },
            _meta: metadata,
          },
        }).then(() => this.send({
          jsonrpc: "2.0",
          method: "notifications/tasks/status",
          params: {
            taskId: taskIds[0],
            status: "working",
            statusMessage: `fixture update ${taskNotificationSecret}`,
            createdAt: "2030-01-01T00:00:00Z",
            lastUpdatedAt: "2030-01-01T00:00:01Z",
            ttlMs: 60_000,
            pollIntervalMs: 0,
            "ui/resourceUri": `ui://fixture/${taskNotificationSecret}`,
            "ui/visibility": ["model"],
            _meta: metadata,
          },
        })).catch(() => {});
        return;
      }
      if (message?.method === "notifications/cancelled" &&
          this.taskSubscriptionIds.delete(message?.params?.requestId)) {
        return;
      }
      if (["tasks/get", "tasks/update", "tasks/cancel"].includes(message?.method)) {
        const handler = taskExtensionHandler;
        if (!handler || message.id === undefined) {
          void this.send({
            jsonrpc: "2.0",
            id: message?.id ?? null,
            error: { code: -32601, message: "Method not found" },
          });
          return;
        }
        void Promise.resolve(handler(message.method, message.params ?? {})).then(
          (result) => this.send({ jsonrpc: "2.0", id: message.id, result }),
          () => this.send({
            jsonrpc: "2.0",
            id: message.id,
            error: { code: -32602, message: "Invalid Tasks extension request" },
          }),
        );
        return;
      }
      sdkMessageHandler?.(message);
    };
    await super.start();
  }
}

const loadTaskState = () => {
  if (!taskStateFile || !fs.existsSync(taskStateFile)) {
    return { counter: 0, tasks: new Map() };
  }
  const parsed = JSON.parse(fs.readFileSync(taskStateFile, "utf8"));
  if (!Number.isInteger(parsed.counter) || !parsed.tasks || Array.isArray(parsed.tasks)) {
    throw new Error("invalid TypeScript MCP fixture Task state");
  }
  return { counter: parsed.counter, tasks: new Map(Object.entries(parsed.tasks)) };
};

const saveTaskState = (counter, tasks) => {
  if (!taskStateFile) return;
  fs.mkdirSync(path.dirname(taskStateFile), { recursive: true });
  const temporary = `${taskStateFile}.tmp`;
  fs.writeFileSync(
    temporary,
    JSON.stringify({ counter, tasks: Object.fromEntries(tasks) }),
    "utf8",
  );
  fs.renameSync(temporary, taskStateFile);
};

serveStdio(() => {
  let revision = 1;
  const persistedTasks = loadTaskState();
  let taskCounter = persistedTasks.counter;
  const tasks = persistedTasks.tasks;
  const TASK_CREATED_AT = "2030-01-01T00:00:00Z";
  const TASK_INITIAL_UPDATED_AT = "2030-01-01T00:00:01Z";
  const TASK_INPUT_UPDATED_AT = "2030-01-01T00:00:02Z";
  const TASK_COMPLETE_UPDATED_AT = "2030-01-01T00:00:03Z";
  const server = new McpServer(
    {
      name: "agent-libos-typescript-sdk-v2-fixture",
      version: "2.0.0-beta.4",
    },
    {
      capabilities: {
        extensions: { "io.modelcontextprotocol/tasks": {} },
        prompts: { listChanged: true },
        resources: { listChanged: true, subscribe: true },
        tools: { listChanged: true },
      },
    },
  );

  server.registerResource(
    "current-document",
    RESOURCE_URI,
    {
      title: "Current fixture document",
      description: "A deterministic text resource used by the Agent libOS MCP gate.",
      mimeType: "text/plain",
    },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/plain",
          text: `typescript-sdk-v2 revision=${revision}`,
        },
      ],
    }),
  );

  server.registerResource(
    "named-document",
    new ResourceTemplate("fixture://document/{name}", { list: undefined }),
    {
      title: "Named fixture document",
      description: "A deterministic resource-template fixture.",
      mimeType: "text/plain",
    },
    async (uri, variables) => ({
      contents: [
        {
          uri: uri.href,
          mimeType: "text/plain",
          text: `typescript-sdk-v2 name=${variables.name}`,
        },
      ],
    }),
  );

  server.registerPrompt(
    "review_document",
    {
      title: "Review fixture document",
      description: "Build a deterministic review prompt.",
      argsSchema: z.object({ focus: z.string().default("correctness") }),
    },
    async ({ focus }) => ({
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Review the fixture document for ${focus}.`,
          },
        },
      ],
    }),
  );

  server.registerTool(
    "publish_resource_update",
    {
      title: "Publish fixture resource update",
      description: "Increment the fixture resource and emit a modern subscription event.",
      inputSchema: z.object({}),
      outputSchema: z.object({ uri: z.string(), revision: z.number().int() }),
    },
    async () => {
      revision += 1;
      await server.server.sendResourceUpdated({ uri: RESOURCE_URI });
      const structuredContent = { uri: RESOURCE_URI, revision };
      return {
        content: [{ type: "text", text: JSON.stringify(structuredContent) }],
        structuredContent,
      };
    },
  );

  const taskResult = (taskId, state, initial) => {
    const lastUpdatedAt = {
      working: TASK_INITIAL_UPDATED_AT,
      input_required: TASK_INPUT_UPDATED_AT,
      completed: TASK_COMPLETE_UPDATED_AT,
      cancelled: TASK_COMPLETE_UPDATED_AT,
    }[state.status];
    const result = {
      resultType: initial ? "task" : "complete",
      taskId,
      status: state.status,
      createdAt: TASK_CREATED_AT,
      lastUpdatedAt,
      ttlMs: 60_000,
      pollIntervalMs: 0,
    };
    if (state.status === "input_required") {
      result.inputRequests = {
        "remote-review-input": {
          method: "elicitation/create",
          params: {
            message: "Approve the fixture review?",
            requestedSchema: {
              type: "object",
              properties: { approved: { type: "boolean" } },
              required: ["approved"],
            },
          },
        },
      };
    } else if (state.status === "completed") {
      result.result = {
        approved: true,
        source: "typescript-sdk-v2-tasks-extension",
      };
    }
    return result;
  };

  server.registerTool(
    "begin_review_task",
    {
      title: "Begin fixture review task",
      description: "Return a digest-pinned Tasks extension handle.",
      inputSchema: z.object({ mode: z.enum(["input", "cancel"]).default("input") }),
    },
    async ({ mode }) => {
      taskCounter += 1;
      const taskId = `typescript-sdk-v2-private-task-${taskCounter}`;
      const state = { mode, status: "working" };
      tasks.set(taskId, state);
      saveTaskState(taskCounter, tasks);
      return taskResult(taskId, state, true);
    },
  );

  // This SDK beta intentionally removed the high-level Tasks execution
  // surface while retaining the released core wire models.  Its generic
  // registerTool wrapper therefore appends `content: []` to an extension
  // CreateTaskResult.  Exercise the independent SDK's audited low-level
  // request-handler seam instead so the fixture emits the official claimed
  // result shape unchanged.  Tool discovery remains owned by registerTool.
  server.server.setRequestHandler(
    "tools/call",
    {
      params: z.object({
        name: z.string(),
        arguments: z.record(z.string(), z.unknown()).optional(),
      }).passthrough(),
      result: z.object({}).passthrough(),
    },
    async ({ name, arguments: rawArguments = {} }) => {
      if (name === "begin_review_task") {
        const mode = rawArguments.mode ?? "input";
        if (!(["input", "cancel"].includes(mode))) {
          throw new Error("invalid fixture Task mode");
        }
        taskCounter += 1;
        const taskId = `typescript-sdk-v2-private-task-${taskCounter}`;
        const state = { mode, status: "working" };
        tasks.set(taskId, state);
        saveTaskState(taskCounter, tasks);
        return taskResult(taskId, state, true);
      }
      if (name === "publish_resource_update") {
        revision += 1;
        await server.server.sendResourceUpdated({ uri: RESOURCE_URI });
        const structuredContent = { uri: RESOURCE_URI, revision };
        return {
          content: [{ type: "text", text: JSON.stringify(structuredContent) }],
          structuredContent,
        };
      }
      throw new Error("unknown fixture Tool");
    },
  );

  const taskReference = z.object({ taskId: z.string().min(1) }).strict();
  const taskUpdate = z.object({
    taskId: z.string().min(1),
    inputResponses: z.record(z.string(), z.unknown()),
  }).strict();
  const taskWireResult = z.object({}).passthrough();
  const emptyResult = z.object({}).strict();

  taskExtensionHandler = async (method, rawParams) => {
    const businessParams = { ...rawParams };
    delete businessParams._meta;
    if (method === "tasks/get") {
      const { taskId } = taskReference.parse(businessParams);
      const state = tasks.get(taskId);
      if (!state) throw new Error("unknown fixture Task");
      if (state.mode === "input" && state.status === "working") {
        state.status = "input_required";
        saveTaskState(taskCounter, tasks);
      }
      return taskResult(taskId, state, false);
    }
    if (method === "tasks/update") {
      const { taskId, inputResponses } = taskUpdate.parse(businessParams);
      const state = tasks.get(taskId);
      if (!state || state.status !== "input_required") {
        throw new Error("fixture Task is not awaiting input");
      }
      const expected = {
        "remote-review-input": {
          action: "accept",
          content: { approved: true },
        },
      };
      if (JSON.stringify(inputResponses) !== JSON.stringify(expected)) {
        throw new Error("fixture Task received an unbound input response");
      }
      state.status = "completed";
      saveTaskState(taskCounter, tasks);
      return { resultType: "complete" };
    }
    if (method === "tasks/cancel") {
      const { taskId } = taskReference.parse(businessParams);
      const state = tasks.get(taskId);
      if (!state || !["working", "input_required"].includes(state.status)) {
        throw new Error("fixture Task cannot be cancelled");
      }
      state.status = "cancelled";
      saveTaskState(taskCounter, tasks);
      return { resultType: "complete" };
    }
    throw new Error("unknown Tasks extension method");
  };

  server.server.setRequestHandler(
    "tasks/get",
    { params: taskReference, result: taskWireResult },
    async ({ taskId }) => {
      const state = tasks.get(taskId);
      if (!state) throw new Error("unknown fixture Task");
      if (state.mode === "input" && state.status === "working") {
        state.status = "input_required";
        saveTaskState(taskCounter, tasks);
      }
      return taskResult(taskId, state, false);
    },
  );
  server.server.setRequestHandler(
    "tasks/update",
    { params: taskUpdate, result: emptyResult },
    async ({ taskId, inputResponses }) => {
      const state = tasks.get(taskId);
      if (!state || state.status !== "input_required") {
        throw new Error("fixture Task is not awaiting input");
      }
      const expected = {
        "remote-review-input": {
          action: "accept",
          content: { approved: true },
        },
      };
      if (JSON.stringify(inputResponses) !== JSON.stringify(expected)) {
        throw new Error("fixture Task received an unbound input response");
      }
      state.status = "completed";
      saveTaskState(taskCounter, tasks);
      return {};
    },
  );
  server.server.setRequestHandler(
    "tasks/cancel",
    { params: taskReference, result: emptyResult },
    async ({ taskId }) => {
      const state = tasks.get(taskId);
      if (!state || !["working", "input_required"].includes(state.status)) {
        throw new Error("fixture Task cannot be cancelled");
      }
      state.status = "cancelled";
      saveTaskState(taskCounter, tasks);
      return {};
    },
  );

  // The modern Runtime gate opens a real `subscriptions/listen` stream but
  // deliberately has no raw SDK handle with which to trigger an event.  The
  // server API has no subscribe callback, so use bounded repeated publication:
  // pre-handshake attempts are harmless and at least one post-handshake event
  // is delivered even on a slow hosted runner.  `unref()` prevents this
  // gate-only timer from extending the fixture process lifetime.
  let fixtureUpdateAttempts = 0;
  const fixtureUpdates = setInterval(() => {
    fixtureUpdateAttempts += 1;
    void server.server.sendResourceUpdated({ uri: RESOURCE_URI }).catch(() => {});
    if (fixtureUpdateAttempts >= 50) {
      clearInterval(fixtureUpdates);
    }
  }, 200);
  fixtureUpdates.unref();

  return server;
}, { transport: new TasksExtensionTransport() });
