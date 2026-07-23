import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, LibOSClient, parseSseFrame } from "./client";

afterEach(() => {
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
    const malformedFetch = mockFetch({ db: "local", scheduler: {}, processes: [] });
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
});

function mockFetch(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue(payload)
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
