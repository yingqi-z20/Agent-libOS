import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  capabilityInventoryMaxItems,
  capabilityInventoryMaxPages,
  isUnadmittedTaskRunRevisionConflict,
  LibOSClient,
  objectTaskWaitDeadlineMarginMs,
  objectTaskWaitDeadlineMs,
  parseSseFrame
} from "./client";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("parseSseFrame", () => {
  it("parses named JSON SSE events", () => {
    const message = parseSseFrame('id: 42\nevent: snapshot\ndata: {"ok": true}\n');
    expect(message).toEqual({ id: "42", event: "snapshot", data: { ok: true } });
  });

  it("ignores invalid JSON frames", () => {
    expect(parseSseFrame("id: 43\nevent: snapshot\ndata: {bad}\n")).toBeNull();
  });
});

describe("LibOSClient", () => {
  it("reports live-stream connection state and stops cleanly when aborted", async () => {
    const controller = new AbortController();
    const statuses: string[] = [];
    const messages: unknown[] = [];
    const body = new ReadableStream<Uint8Array>({
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('id: 1\r\nevent: snapshot\r\ndata: {"snapshot": null}\r\n\r\n'));
        streamController.close();
      }
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200, body }));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.stream((message) => {
      messages.push(message);
      controller.abort();
    }, controller.signal, "0", (status) => statuses.push(status));

    expect(statuses).toEqual(["connecting", "connected"]);
    expect(messages).toEqual([{ id: "1", event: "snapshot", data: { snapshot: null } }]);
  });

  it("marks terminal SSE HTTP failures instead of silently reconnecting", async () => {
    const statuses: string[] = [];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 401, body: null }));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.stream(() => undefined, new AbortController().signal, "0", (status) => statuses.push(status))).rejects.toThrow(/401/);
    expect(statuses).toEqual(["connecting", "failed"]);
  });

  it("passes task authority, initial working directory, and LLM profile through spawn requests", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    const authorityManifest = {
      authorized_capabilities: [{ resource: "human:owner", rights: ["write"] }],
      approval_policy: {
        requestable_capabilities: [
          { resource: "filesystem:workspace:src/app/*", rights: ["read", "write"] }
        ]
      }
    };

    await client.spawn("goal", "coding-agent:v0", 4, false, {
      authorityManifest,
      workingDirectory: " src/app ",
      llmProfile: "qwen3.7-max"
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes",
      expect.objectContaining({
        body: JSON.stringify({
          goal: "goal",
          image: "coding-agent:v0",
          auto_run: false,
          llm_profile: "qwen3.7-max",
          working_directory: "src/app",
          authority_manifest: authorityManifest,
          max_quanta: 4
        })
      })
    );
  });

  it("passes max_quanta and LLM profile through exec requests", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.execProcess("pid_1", "base-agent:v0", "goal", true, false, 7, "glm-5.2");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes/pid_1/exec",
      expect.objectContaining({
        body: JSON.stringify({
          image: "base-agent:v0",
          goal: "goal",
          confirmed: true,
          auto_run: false,
          llm_profile: "glm-5.2",
          max_quanta: 7
        })
      })
    );
  });

  it("manages user LLM profiles through the GUI API", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.createLLMProfile({ profile_id: "kimi-k2.7-code", model: "kimi-k2.7-code", api_key_env: "KIMI_API_KEY" });
    await client.updateLLMProfile("kimi-k2.7-code", { model: "kimi-k2.7-code", api_key_env: "KIMI_API_KEY", api_mode: "chat" });
    await client.deleteLLMProfile("kimi-k2.7-code");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:1/api/llm-profiles",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ profile_id: "kimi-k2.7-code", model: "kimi-k2.7-code", api_key_env: "KIMI_API_KEY" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:1/api/llm-profiles/kimi-k2.7-code",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ model: "kimi-k2.7-code", api_key_env: "KIMI_API_KEY", api_mode: "chat" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://127.0.0.1:1/api/llm-profiles/kimi-k2.7-code",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  it("can explicitly confirm a high-risk workflow request", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.runWorkflow({
      tool: "write_text_file",
      args: { path: "result.txt", content: "done" },
      confirmed: true
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/workflows/run",
      expect.objectContaining({
        body: JSON.stringify({
          tool: "write_text_file",
          args: { path: "result.txt", content: "done" },
          confirmed: true
        })
      })
    );
  });

  it("passes typed permission decisions and scheduler options through human responses", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.respondHumanRequest(
      "request_1",
      { kind: "permission", approved: true, decision: { policy: "ask_each_time" } },
      false,
      3
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/human-requests/request_1/respond",
      expect.objectContaining({
        body: JSON.stringify({
          approved: true,
          decision: { policy: "ask_each_time" },
          auto_run: false,
          max_quanta: 3
        })
      })
    );
  });

  it("passes typed question answers without inventing a permission decision", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.respondHumanRequest("request_2", { kind: "question", approved: true, answer: "eu-west" }, true, null);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/human-requests/request_2/respond",
      expect.objectContaining({
        body: JSON.stringify({
          approved: true,
          answer: "eu-west",
          auto_run: true
        })
      })
    );
  });

  it("submits agent ratings for the selected process", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.submitAgentRating("pid_1", 4, "good result");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes/pid_1/rating",
      expect.objectContaining({
        body: JSON.stringify({
          score: 4,
          comment: "good result"
        })
      })
    );
  });

  it("preserves explicit confirmations and authority mode for administration routes", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.restoreCheckpoint("cp/1", true, "pid_1");
    await client.registerSkill("skills/reviewer", "pid_1", true, true);
    await client.grantCapability({ subject: "pid_1", resource: "object:report", rights: ["read"] }, true);
    await client.callJsonRpc("endpoint/1", "pid_1", "lookup", { key: "x" }, true);
    await client.callMcpTool("server/1", "pid_1", "search", { q: "x" }, true);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:1/api/checkpoints/cp%2F1/restore",
      expect.objectContaining({ body: JSON.stringify({ confirmed: true, actor: "pid_1" }) })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:1/api/skills/register",
      expect.objectContaining({ body: JSON.stringify({ path: "skills/reviewer", actor: "pid_1", confirmed: true, replace: true }) })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://127.0.0.1:1/api/capabilities/grant",
      expect.objectContaining({ body: JSON.stringify({ subject: "pid_1", resource: "object:report", rights: ["read"], confirmed: true }) })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      "http://127.0.0.1:1/api/jsonrpc/endpoint%2F1/call",
      expect.objectContaining({ body: JSON.stringify({ pid: "pid_1", method_id: "lookup", params: { key: "x" }, confirmed: true }) })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      "http://127.0.0.1:1/api/mcp/server%2F1/call",
      expect.objectContaining({ body: JSON.stringify({ pid: "pid_1", tool_id: "search", arguments: { q: "x" }, confirmed: true }) })
    );
  });

  it("passes the discovered package hash through skill activation", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    const packageSha256 = "b".repeat(64);

    await client.activateSkill("reviewer/1", "pid_1", packageSha256, true, "pid_1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/skills/reviewer%2F1/activate",
      expect.objectContaining({
        body: JSON.stringify({
          pid: "pid_1",
          expected_package_sha256: packageSha256,
          confirmed: true,
          actor: "pid_1"
        })
      })
    );
  });

  it("does not add a content-type header to read-only GET requests", async () => {
    const fetchMock = mockFetch({ ok: true });
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.health();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/health",
      expect.objectContaining({
        method: "GET",
        headers: { Authorization: "Bearer token", Accept: "application/json" }
      })
    );
  });

  it("rejects malformed snapshots and retains structured API error context", async () => {
    const malformedFetch = mockFetch({ schema_version: 2, db: "local", scheduler: {}, processes: [] });
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    await expect(client.snapshot()).rejects.toThrow(/scheduler/);
    expect(malformedFetch).toHaveBeenCalledTimes(1);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 403,
      json: vi.fn().mockResolvedValue({ error: { message: "denied", details: { resource: "object:x" } } })
    }));
    const request = client.request("GET", "/api/capabilities");
    await expect(request).rejects.toBeInstanceOf(ApiError);
    await expect(request).rejects.toMatchObject({ status: 403, message: "denied" });
  });

  it("supports a per-call requestJson deadline", async () => {
    vi.stubGlobal("fetch", vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    })));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.requestJson("GET", "/api/snapshot", undefined, { timeoutMs: 5 })).rejects.toMatchObject({ name: "TimeoutError" });
  });

  it("keeps the default read deadline while leaving non-idempotent mutations unbounded", async () => {
    vi.useFakeTimers();
    const signals: AbortSignal[] = [];
    vi.stubGlobal("fetch", vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init.signal as AbortSignal;
      signals.push(signal);
      signal.addEventListener("abort", () => reject(signal.reason), { once: true });
    })));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    const readResult = client.requestJson("GET", "/api/snapshot").catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(29_999);
    expect(signals[0].aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await expect(readResult).resolves.toMatchObject({ name: "TimeoutError" });

    const mutationAbort = new AbortController();
    const mutationResult = client.requestJson(
      "POST",
      "/api/processes",
      { goal: "slow mutation" },
      { signal: mutationAbort.signal }
    ).catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(30_001);
    expect(signals[1].aborted).toBe(false);
    mutationAbort.abort(new DOMException("cancelled by caller", "AbortError"));
    await expect(mutationResult).resolves.toMatchObject({ name: "AbortError" });
  });

  it("gives explicit object-task waits a transport margin without guessing active server config", async () => {
    expect(objectTaskWaitDeadlineMs()).toBeNull();
    expect(objectTaskWaitDeadlineMs(1.25)).toBe(1_250 + objectTaskWaitDeadlineMarginMs);
    expect(objectTaskWaitDeadlineMs(300)).toBe(300_000 + objectTaskWaitDeadlineMarginMs);
    expect(objectTaskWaitDeadlineMs(400)).toBe(400_000 + objectTaskWaitDeadlineMarginMs);
    expect(() => objectTaskWaitDeadlineMs(-1)).toThrow(/finite non-negative/);

    const fetchMock = mockFetch({ task_id: "task_1", status: "running" });
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    await client.waitObjectTask("task_1", "pid_1", 400);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/object-tasks/task_1/wait",
      expect.objectContaining({ body: JSON.stringify({ pid: "pid_1", timeout_s: 400 }) })
    );
  });

  it("continues process audit pagination with the opaque before cursor", async () => {
    const fetchMock = mockFetch([]);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.listProcessAudit("pid/1", 50, "audit_cursor");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes/pid%2F1/audit?limit=50&before=audit_cursor",
      expect.objectContaining({ method: "GET" })
    );
  });

  it("omits the process-audit limit so the active server config selects the page size", async () => {
    const fetchMock = mockFetch([]);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.listProcessAudit("pid/1", undefined, "audit_cursor");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes/pid%2F1/audit?before=audit_cursor",
      expect.objectContaining({ method: "GET" })
    );
  });

  it("walks the paginated capability endpoint until the complete inventory is loaded", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        items: [capability("cap_1")],
        next_after: "cap_1",
        has_more: true
      }))
      .mockResolvedValueOnce(jsonResponse({
        items: [capability("cap_2")],
        next_after: null,
        has_more: false
      }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.listCapabilities("pid/1")).resolves.toEqual([
      capability("cap_1"),
      capability("cap_2")
    ]);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:1/api/capabilities?mode=page&subject=pid%2F1",
      expect.objectContaining({ method: "GET" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:1/api/capabilities?mode=page&subject=pid%2F1&after=cap_1",
      expect.objectContaining({ method: "GET" })
    );
  });

  it("bounds capability inventory traversal by both page and item count", async () => {
    const pageBoundClient = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    let page = 0;
    vi.spyOn(pageBoundClient, "listCapabilityPage").mockImplementation(async () => {
      page += 1;
      return { items: [capability(`cap_${page}`)], next_after: `cursor_${page}`, has_more: true };
    });
    await expect(pageBoundClient.listCapabilities("pid_1")).rejects.toThrow(
      `exceeds ${capabilityInventoryMaxPages} pages`
    );
    expect(page).toBe(capabilityInventoryMaxPages);

    const itemBoundClient = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });
    vi.spyOn(itemBoundClient, "listCapabilityPage").mockResolvedValue({
      items: Array.from({ length: capabilityInventoryMaxItems + 1 }, (_, index) => capability(`cap_${index}`)),
      next_after: null,
      has_more: false
    });
    await expect(itemBoundClient.listCapabilities("pid_1")).rejects.toThrow(
      `exceeds ${capabilityInventoryMaxItems} items`
    );
  });

  it("reconnects SSE from the last delivered id when the stream fails mid-read", async () => {
    const controller = new AbortController();
    let reads = 0;
    const firstBody = new ReadableStream<Uint8Array>({
      pull(streamController) {
        if (reads++ === 0) {
          streamController.enqueue(new TextEncoder().encode('id: 7\nevent: snapshot\ndata: {"snapshot":null}\n\n'));
        } else {
          streamController.error(new Error("connection reset"));
        }
      }
    });
    const urls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      urls.push(url);
      if (urls.length === 1) return { ok: true, status: 200, body: firstBody };
      controller.abort();
      throw new Error("aborted reconnect");
    }));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.stream(() => undefined, controller.signal);

    expect(urls[1]).toContain("cursor=7");
  });

  it("uses revision-fenced, idempotent durable run mutations and explicit confirmations", async () => {
    const fetchMock = mockFetch(taskRunSummary());
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.createTaskRun({
      schema_version: 1,
      goal: "finish",
      display_title: "Finish",
      image_id: "coding-agent:v0",
      retention: "purge_on_terminal"
    }, "create-1");
    await client.runTaskRun("run/1", 4, "run-1", 8);
    await client.cancelTaskRun("run/1", 4, "cancel-1", true, "stop");
    await client.followUpTaskRun("run/1", "durable follow-up", 4, "follow-up-1", {
      kind: "interrupt",
      required: true
    });
    await client.recoverTaskRun("run/1", "register-receipt", 4, "recover-1", true, { receipt_id: "r1" });
    await client.rerunTaskRun("run/1", 4, "rerun-1", {
      specOverrides: { goal: "replacement goal" }
    });

    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://127.0.0.1:1/api/task-runs", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        spec: { schema_version: 1, goal: "finish", display_title: "Finish", image_id: "coding-agent:v0", retention: "purge_on_terminal" },
        client_request_id: "create-1"
      })
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "http://127.0.0.1:1/api/task-runs/run%2F1/run", expect.objectContaining({
      body: JSON.stringify({ expected_revision: 4, command_id: "run-1", max_quanta: 8 })
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "http://127.0.0.1:1/api/task-runs/run%2F1/cancel", expect.objectContaining({
      body: JSON.stringify({ expected_revision: 4, command_id: "cancel-1", confirmed: true, reason: "stop" })
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(4, "http://127.0.0.1:1/api/task-runs/run%2F1/follow-ups", expect.objectContaining({
      body: JSON.stringify({
        expected_revision: 4,
        command_id: "follow-up-1",
        body: "durable follow-up",
        kind: "interrupt",
        required: true
      })
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(5, "http://127.0.0.1:1/api/task-runs/run%2F1/recover", expect.objectContaining({
      body: JSON.stringify({ expected_revision: 4, command_id: "recover-1", option_id: "register-receipt", confirmed: true, receipt: { receipt_id: "r1" } })
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(6, "http://127.0.0.1:1/api/task-runs/run%2F1/rerun", expect.objectContaining({
      body: JSON.stringify({
        expected_revision: 4,
        command_id: "rerun-1",
        client_request_id: "rerun-1:create",
        spec_overrides: { goal: "replacement goal" }
      })
    }));
  });

  it("sends durable pause without an unsupported reason field", async () => {
    const fetchMock = mockFetch(taskRunSummary());
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.pauseTaskRun("run/1", 4, "pause-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/task-runs/run%2F1/pause",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ expected_revision: 4, command_id: "pause-1" })
      })
    );
  });

  it("reconciles a stable TaskRun 409 through an exact detail read without retrying the mutation", async () => {
    const conflictSummary = { ...taskRunSummary(), revision: 5 };
    const latestSummary = { ...taskRunSummary(), revision: 6 };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorJsonResponse(409, {
        ok: false,
        error: {
          type: "TaskRunRevisionConflict",
          code: "task_run_revision_conflict",
          message: "stale TaskRun revision",
          command_admitted: false,
          current_summary: conflictSummary
        }
      }))
      .mockResolvedValueOnce(jsonResponse({
        summary: latestSummary,
        requirements: { items: [], next_cursor: null, has_more: false },
        recovery_options: []
      }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    let rejected: unknown;
    try {
      await client.pauseTaskRun("run_1", 4, "pause-stale");
    } catch (error) {
      rejected = error;
    }

    expect(rejected).toBeInstanceOf(ApiError);
    expect(isUnadmittedTaskRunRevisionConflict(rejected)).toBe(true);
    expect((rejected as { currentSummary?: unknown }).currentSummary).toEqual(latestSummary);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://127.0.0.1:1/api/task-runs/run_1/pause",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ expected_revision: 4, command_id: "pause-stale" })
      })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://127.0.0.1:1/api/task-runs/run_1",
      expect.objectContaining({ method: "GET" })
    );
  });

  it("never rotates a command conflict or an admitted revision conflict", () => {
    const commandConflict = new ApiError("conflict", 409, {
      error: { code: "task_run_command_conflict", command_admitted: true }
    });
    const admittedRevision = new ApiError("conflict", 409, {
      error: { code: "task_run_revision_conflict", command_admitted: true }
    });

    expect(isUnadmittedTaskRunRevisionConflict(commandConflict)).toBe(false);
    expect(isUnadmittedTaskRunRevisionConflict(admittedRevision)).toBe(false);
  });

  it("uses exact human-request reconciliation and authorized run-detail pages", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        summary: { ...taskRunSummary(), run_id: "run/1" },
        requirements: { items: [], next_cursor: null, has_more: false },
        recovery_options: []
      }))
      .mockResolvedValueOnce(jsonResponse({ items: [], next_cursor: null, has_more: false }))
      .mockResolvedValueOnce(jsonResponse({
        items: [{
          request_id: "human/1",
          pid: "pid_1",
          human: "owner",
          payload: { type: "approval" },
          status: "pending",
          decision: null,
          blocking: true,
          created_at: "2030-01-01T00:00:00Z",
          updated_at: "2030-01-01T00:00:00Z"
        }],
        next_cursor: "human-cursor/2",
        has_more: true,
        presentation_truncated: false
      }))
      .mockResolvedValueOnce(jsonResponse({
        request_id: "human/1",
        pid: "pid_1",
        human: "owner",
        payload: { type: "approval" },
        status: "pending",
        decision: null,
        blocking: true,
        created_at: "2030-01-01T00:00:00Z",
        updated_at: "2030-01-01T00:00:00Z"
      }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.getTaskRun("run/1");
    await client.listTaskRunLedger("run/1", 50, "cursor/1");
    await client.listTaskRunHumanRequests("run/1", 25, "human-cursor/1", ["pending"]);
    await client.getHumanRequest("human/1");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://127.0.0.1:1/api/task-runs/run%2F1", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "http://127.0.0.1:1/api/task-runs/run%2F1/ledger?limit=50&cursor=cursor%2F1", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "http://127.0.0.1:1/api/task-runs/run%2F1/human-requests?limit=25&cursor=human-cursor%2F1&status=pending", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(4, "http://127.0.0.1:1/api/human-requests/human%2F1", expect.objectContaining({ method: "GET" }));
  });

  it("rejects an exact Human response with a mismatched identity", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      request_id: "human-other",
      pid: "pid_1",
      human: "owner",
      payload: { type: "approval" },
      status: "pending",
      decision: null,
      blocking: true,
      created_at: "2030-01-01T00:00:00Z",
      updated_at: "2030-01-01T00:00:00Z"
    })));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.getHumanRequest("human-requested")).rejects.toThrow(
      "identity does not match"
    );
  });
});

function mockFetch(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function jsonResponse(payload: unknown) {
  return {
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue(payload)
  };
}

function errorJsonResponse(status: number, payload: unknown) {
  return {
    ok: false,
    status,
    json: vi.fn().mockResolvedValue(payload)
  };
}

function capability(capId: string) {
  return {
    cap_id: capId,
    subject: "pid/1",
    resource: `object:${capId}`,
    rights: ["read"]
  };
}

function taskRunSummary() {
  return {
    schema_version: 1,
    run_id: "run_1",
    revision: 4,
    status: "paused",
    display_title: "Finish",
    root_pid: "pid_1",
    active_pid: "pid_1",
    allowed_actions: ["resume", "cancel", "recover"],
    blockers: [],
    retention: "purge_on_terminal",
    payloads_purged: false
  };
}
