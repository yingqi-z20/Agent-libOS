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

  it("binds external-operation responses to the canonical preview revision and digest", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.respondHumanRequest(
      "request_external",
      {
        kind: "external_approval",
        approved: true,
        expected_revision: 7,
        preview_sha256: "a".repeat(64)
      },
      false,
      null
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/human-requests/request_external/respond",
      expect.objectContaining({
        body: JSON.stringify({
          approved: true,
          expected_revision: 7,
          preview_sha256: "a".repeat(64),
          auto_run: false
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
    const fetchMock = mockFetch({
      server_id: "server/1",
      tool_id: "search",
      mcp_name: "search",
      status: "ok",
      ok: true,
      result: null,
      error: null,
      response_bytes: 0,
      duration_s: 0,
      connection: null,
      receipts: []
    });
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

  it("discovers MCP protocol state through the encoded Host route without confirmation", async () => {
    const payload = {
      server_id: "server/1",
      connection: {
        protocol_mode: "auto",
        protocol_era: "modern",
        protocol_revision: "2026-07-28",
        sessionless: true,
        fallback_used: false,
        capabilities: ["tools"],
        unsupported_capabilities: []
      },
      request_bytes: 64,
      response_bytes: 96,
      duration_s: 0.01,
      receipts: [{
        phase: "server/discover",
        request_bytes: 64,
        response_bytes: 96,
        duration_s: 0.01,
        call_started: true
      }]
    };
    const fetchMock = mockFetch(payload);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.discoverMcpServer("server/1")).resolves.toEqual(payload);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/mcp/server%2F1/discover",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({})
      })
    );
    expect(JSON.parse(fetchMock.mock.calls[0]?.[1]?.body as string)).not.toHaveProperty("confirmed");
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

    await client.images();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/images",
      expect.objectContaining({
        method: "GET",
        headers: { Authorization: "Bearer token", Accept: "application/json" }
      })
    );
  });

  it("uses process-bound cursor routes for LLM trace list, detail, and content", async () => {
    const call = llmCallSummaryFixture("pid/1", "call/1");
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [call], next_cursor: "next/list", has_more: true }))
      .mockResolvedValueOnce(jsonResponse(llmCallDetailFixture(call)))
      .mockResolvedValueOnce(jsonResponse({
        schema_version: 1,
        pid: "pid/1",
        call_id: "call/1",
        field: "attempt_reasoning",
        attempt_sequence: 1,
        content: "reasoning",
        next_cursor: null,
        has_more: false,
        content_hash: "a".repeat(64),
        retention_tier: "full"
      }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.listProcessLlmCalls("pid/1", 25, "before/1");
    await client.getProcessLlmCall("pid/1", "call/1");
    await client.getProcessLlmCallContent("pid/1", "call/1", "attempt_reasoning", {
      attemptSequence: 1,
      cursor: "signed/0",
      limit: 32_768
    });

    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://127.0.0.1:1/api/processes/pid%2F1/llm-calls?limit=25&cursor=before%2F1", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "http://127.0.0.1:1/api/processes/pid%2F1/llm-calls/call%2F1", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, "http://127.0.0.1:1/api/processes/pid%2F1/llm-calls/call%2F1/content?field=attempt_reasoning&limit=32768&attempt_sequence=1&cursor=signed%2F0", expect.objectContaining({ method: "GET" }));
  });

  it("rejects malformed snapshots and retains structured API error context", async () => {
    const malformedFetch = mockFetch({ schema_version: 3, db: "local", scheduler: {}, processes: [] });
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

  it("supports a per-call request deadline", async () => {
    vi.stubGlobal("fetch", vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    })));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.request("GET", "/api/snapshot", undefined, { timeoutMs: 5 })).rejects.toMatchObject({ name: "TimeoutError" });
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

    const readResult = client.request("GET", "/api/snapshot").catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(29_999);
    expect(signals[0].aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await expect(readResult).resolves.toMatchObject({ name: "TimeoutError" });

    const mutationAbort = new AbortController();
    const mutationResult = client.request(
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
          revision: 0,
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
        revision: 0,
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
      revision: 0,
      created_at: "2030-01-01T00:00:00Z",
      updated_at: "2030-01-01T00:00:00Z"
    })));
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.getHumanRequest("human-requested")).rejects.toThrow(
      "identity does not match"
    );
  });

  it("loads only bounded read-only Semantic status, pages, and exact details", async () => {
    const summary = semanticSummaryFixture();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(semanticStatusFixture()))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [summary], next_cursor: "cursor/2" }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, assessment: semanticDetailFixture() }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.getSemanticStatus();
    await client.listSemanticAssessments({
      pid: "pid/1",
      requestId: "request/1",
      operationId: "operation/1",
      kind: "approval",
      status: "success",
      domain: "filesystem",
      actionId: "filesystem.read",
      tenantBucketSha256: "6".repeat(64)
    }, 25, "cursor/1");
    await client.getSemanticAssessment("assessment/1");

    expect(fetchMock).toHaveBeenNthCalledWith(1, "http://127.0.0.1:1/api/semantic/status", expect.objectContaining({ method: "GET" }));
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `http://127.0.0.1:1/api/semantic/assessments?pid=pid%2F1&request_id=request%2F1&operation_id=operation%2F1&kind=approval&status=success&domain=filesystem&action_id=filesystem.read&tenant_bucket_sha256=${"6".repeat(64)}&after=cursor%2F1&limit=25`,
      expect.objectContaining({ method: "GET" })
    );
    expect(fetchMock).toHaveBeenNthCalledWith(3, "http://127.0.0.1:1/api/semantic/assessments/assessment%2F1", expect.objectContaining({ method: "GET" }));
  });

  it("rejects malformed Semantic projections and mismatched identities", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [{ ...semanticSummaryFixture(), prompt: "secret" }], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, assessment: { ...semanticDetailFixture(), assessment_id: "other" } }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.listSemanticAssessments()).rejects.toThrow(/private field/);
    await expect(client.getSemanticAssessment("assessment/1")).rejects.toThrow(/identity does not match/);
  });

  it("uses only bounded GET surfaces for Flow, settlement, epoch, control, health, and canary evidence", async () => {
    const flow = semanticStatusFixture().flow;
    const control = {
      schema_version: 1,
      revision: 0,
      generation: 0,
      mode: "off",
      active_epoch_id: null,
      active_policy_sha256: null,
      tripped: false,
      trip_code: null,
      updated_at: "2030-01-01T00:00:00Z"
    };
    const metrics = {
      schema_version: 1,
      window: "7d",
      action_id: "filesystem.read",
      tenant_bucket_sha256: "6".repeat(64),
      epoch_id: "epoch_1",
      risk: "low",
      machine: { ...semanticStatusFixture().machine, eligible: 2, issued: 1 },
      actual_auto_approval: { numerator: 1, denominator: 2, rate: 0.5 },
      review_metrics: {
        reviewed: 2,
        safe: 2,
        unsafe: 0,
        unsafe_rate: 0,
        issued_reviewed: 0,
        issued_review_rate: 0
      }
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(jsonResponse(flow))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse({
        schema_version: 1,
        root_node_id: "entity_1",
        direction: "upstream",
        items: [],
        effective_labels: null,
        coverage: "unknown",
        next_cursor: null,
        truncated: false
      }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse(control))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [control], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse({ schema_version: 1, items: [], next_cursor: null }))
      .mockResolvedValueOnce(jsonResponse(metrics));
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.getSemanticFlowStatus();
    await client.listSemanticFlowEntities({ pid: "pid_1" }, 25, "entity_cursor");
    await client.listSemanticFlowEdges({ pid: "pid_1" }, 25, "edge_cursor");
    await client.getSemanticFlowLineage("entity_1", "upstream", 25, "lineage_cursor");
    await client.listSemanticSettlements({ pid: "pid_1", outcome: "issued" }, 25, "settlement_cursor");
    await client.listSemanticPolicyEpochs(25, "epoch_cursor");
    await client.getSemanticControl();
    await client.listSemanticControlHistory(25, "control_cursor");
    await client.listSemanticHealthEvents(25, "health_cursor");
    await client.getSemanticMetrics({
      window: "7d",
      actionId: "filesystem.read",
      tenantBucketSha256: "6".repeat(64),
      epochId: "epoch_1",
      risk: "low"
    });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:1/api/semantic/flow/status",
      "http://127.0.0.1:1/api/semantic/flow/entities?limit=25&after=entity_cursor&pid=pid_1",
      "http://127.0.0.1:1/api/semantic/flow/edges?limit=25&after=edge_cursor&pid=pid_1",
      "http://127.0.0.1:1/api/semantic/flow/lineage/entity_1?limit=25&after=lineage_cursor&direction=upstream",
      "http://127.0.0.1:1/api/semantic/settlements?limit=25&after=settlement_cursor&pid=pid_1&outcome=issued",
      "http://127.0.0.1:1/api/semantic/policy/epochs?limit=25&after=epoch_cursor",
      "http://127.0.0.1:1/api/semantic/control",
      "http://127.0.0.1:1/api/semantic/control/history?limit=25&after=control_cursor",
      "http://127.0.0.1:1/api/semantic/health?limit=25&after=health_cursor",
      `http://127.0.0.1:1/api/semantic/metrics?action_id=filesystem.read&tenant_bucket_sha256=${"6".repeat(64)}&epoch_id=epoch_1&risk=low&window=7d`
    ]);
    expect(fetchMock.mock.calls.every(([, init]) => init.method === "GET")).toBe(true);
  });

  it("rejects unbounded Semantic pagination before issuing a request", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await expect(client.listSemanticAssessments({}, 0)).rejects.toThrow(/1 to 100/);
    await expect(client.listSemanticAssessments({}, 101)).rejects.toThrow(/1 to 100/);
    await expect(client.listSemanticAssessments({ pid: "" })).rejects.toThrow(/pid/);
    await expect(client.listSemanticAssessments({ kind: "permission" as never })).rejects.toThrow(/kind/);
    await expect(client.listSemanticAssessments({ actionId: "filesystem" })).rejects.toThrow(/action_id/);
    await expect(client.listSemanticAssessments({ tenantBucketSha256: "A".repeat(64) })).rejects.toThrow(/tenant_bucket_sha256/);
    await expect(client.listSemanticAssessments({}, 50, "x".repeat(2_049))).rejects.toThrow(/after/);
    await expect(client.getSemanticAssessment(" ")).rejects.toThrow(/assessment_id/);
    await expect(client.getSemanticMetrics({ risk: "unknown" as never })).rejects.toThrow(/risk/);
    await expect(client.getSemanticFlowLineage("node with spaces", "upstream")).rejects.toThrow(/node_id/);
    expect(fetchMock).not.toHaveBeenCalled();
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

function semanticStatusFixture() {
  return {
    schema_version: 3,
    mode: "shadow",
    adapter: "deterministic",
    profile_id: null,
    queue: { queued: 1, leased: 0, succeeded: 2, failed: 0, cancelled: 0, capture_failures: 0 },
    assessments: {
      total: 2,
      success: 2,
      error: 0,
      ood: 0,
      would_issue_exact_once: 1,
      would_deny: 0,
      require_human: 1,
      by_status: {
        success: 2,
        skipped_policy: 0,
        egress_blocked: 0,
        timeout: 0,
        provider_error: 0,
        provider_outcome_unknown: 0,
        invalid_schema: 0,
        ood: 0,
        abstained: 0,
        stale_input: 0
      },
      by_domain: {
        filesystem: 2,
        shell: 0,
        git: 0,
        jsonrpc: 0,
        mcp: 0,
        runtime: 0,
        unknown: 0
      }
    },
    control: {
      catalog_version: null,
      active_epoch_id: null,
      active_epoch_sha256: null,
      generation: 0,
      state: "inactive",
      trip_reason_code: null
    },
    flow: {
      schema_version: 1,
      available: false,
      counts: { entities: 0, activities: 0, edges: 0, label_assertions: 0 },
      coverage: { complete: 0, partial: 0, unknown: 0, conflict: 0, stale: 0 },
      capture_failures: 0,
      legacy_history: {
        present: false,
        source_schema_version: null,
        assessment_count: 0,
        coverage: null,
        evidence_sha256: null,
        created_at: null
      }
    },
    machine: {
      eligible: 0,
      issued: 0,
      consumed: 0,
      succeeded: 0,
      failed: 0,
      unknown: 0,
      expired: 0,
      revoked: 0,
      race_lost: 0,
      denied: 0
    },
    actual_auto_approval: { numerator: 0, denominator: 0, rate: null },
    review_metrics: {
      reviewed: 0,
      safe: 0,
      unsafe: 0,
      unsafe_rate: null,
      issued_reviewed: 0,
      issued_review_rate: null
    }
  };
}

function semanticSummaryFixture() {
  return {
    assessment_id: "assessment/1",
    job_id: "job/1",
    kind: "approval",
    status: "success",
    domain: "filesystem",
    action_id: "filesystem.read",
    pid: "pid/1",
    request_id: "request/1",
    operation_id: "operation/1",
    effect_id: "effect/1",
    shadow_outcome: "require_human",
    reason_codes: ["missing_authoritative_predicate"],
    ood: false,
    abstain: false,
    confidence_bps: 8000,
    calibration_bucket: "high",
    input_tokens: 120,
    output_tokens: 20,
    cost_microunits: 45,
    classifier_id: "scripted",
    classifier_version: "v1",
    artifact_sha256: "a".repeat(64),
    input_sha256: "b".repeat(64),
    feature_snapshot_sha256: "c".repeat(64),
    policy_sha256: "d".repeat(64),
    created_at: "2030-01-01T00:00:00Z",
    completed_at: "2030-01-01T00:00:01Z",
    latency_ms: 1000,
    human_outcome: null,
    tenant_bucket_sha256: "6".repeat(64)
  };
}

function semanticDetailFixture() {
  return {
    ...semanticSummaryFixture(),
    findings: [{
      code: "missing_authoritative_predicate",
      severity: "medium",
      confidence_bps: 8000,
      evidence_sha256: "e".repeat(64),
      source: "model"
    }],
    data_findings: [],
    matched_rule_ids: [],
    proven_predicates: [],
    missing_predicates: ["low_risk"],
    source_refs_sha256: "1".repeat(64),
    data_labels_sha256: "2".repeat(64),
    sink_identity_sha256: "3".repeat(64),
    tool_schema_sha256: "4".repeat(64),
    provider_spec_sha256: "5".repeat(64),
    manifest_sha256: "7".repeat(64),
    action_sha256: "8".repeat(64),
    resource_sha256: "9".repeat(64),
    args_sha256: "0".repeat(64),
    state_sha256: "a".repeat(64),
    projection_sha256: "b".repeat(64)
  };
}

function llmCallSummaryFixture(pid = "pid_1", callId = "call_1") {
  return {
    schema_version: 1,
    call_id: callId,
    pid,
    image_id: "coding-agent:v0",
    purpose: "agent_loop",
    status: "ok",
    api: "responses",
    model: "test-model",
    usage: {},
    error: null,
    created_at: "2030-01-01T00:00:00Z",
    completed_at: "2030-01-01T00:00:01Z",
    request_id: "req_1",
    response_id: "resp_1",
    attempt_count: 1,
    coverage: "complete",
    selected_attempt: 1,
    reasoning_availability: "returned",
    payload_retention_tier: "full"
  };
}

function llmCallDetailFixture(call: ReturnType<typeof llmCallSummaryFixture>) {
  return {
    schema_version: 1,
    call,
    attempts: [{
      sequence: 1,
      kind: "initial",
      api: "responses",
      status: "ok",
      model: "test-model",
      request_id: "req_1",
      response_id: "resp_1",
      reasoning_availability: "returned",
      reasoning_blocks: [],
      output_availability: "returned",
      tool_names: [],
      tool_call_count: 0,
      usage: {},
      started_at: "2030-01-01T00:00:00Z",
      completed_at: "2030-01-01T00:00:01Z",
      duration_ms: 1000,
      error: null
    }],
    content: [{
      field: "attempt_reasoning",
      attempt_sequence: 1,
      availability: "available",
      content_type: "text",
      size_bytes: 9,
      size_chars: 9,
      content_hash: "a".repeat(64),
      cursor: "signed/0"
    }]
  };
}
