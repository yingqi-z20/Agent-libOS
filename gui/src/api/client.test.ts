import { afterEach, describe, expect, it, vi } from "vitest";
import { LibOSClient, parseSseFrame } from "./client";

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
  it("passes initial working directory and LLM profile through spawn requests", async () => {
    const fetchMock = mockFetch({});
    const client = new LibOSClient({ url: "http://127.0.0.1:1", token: "token", db: "local" });

    await client.spawn("goal", "coding-agent:v0", 4, false, { workingDirectory: " src/app ", llmProfile: "qwen3.7-max" });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:1/api/processes",
      expect.objectContaining({
        body: JSON.stringify({
          goal: "goal",
          image: "coding-agent:v0",
          auto_run: false,
          llm_profile: "qwen3.7-max",
          working_directory: "src/app",
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
});

function mockFetch(payload: unknown) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: vi.fn().mockResolvedValue(payload)
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
