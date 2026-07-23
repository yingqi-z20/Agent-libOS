import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { mergeDotenvIntoEnv, readDotenv, redactGuiServerOutput, requireLoopbackDevServerUrl, runtimeServerEnv } from "./env.js";

const roots: string[] = [];

function tempRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-libos-env-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length > 0) {
    fs.rmSync(roots.pop()!, { recursive: true, force: true });
  }
});

describe("readDotenv", () => {
  it("parses the repo .env format used by the Python launcher", () => {
    const root = tempRoot();
    const envPath = path.join(root, ".env");
    fs.writeFileSync(
      envPath,
      [
        "# comment",
        "OPENAI_API_KEY='from-env-file'",
        'OPENAI_MODEL="gpt-test"',
        "export OPENAI_BASE_URL=https://api.example.test/v1",
        "MALFORMED",
        ""
      ].join("\n"),
      "utf8"
    );

    expect(readDotenv(envPath)).toEqual({
      OPENAI_API_KEY: "from-env-file",
      OPENAI_MODEL: "gpt-test",
      OPENAI_BASE_URL: "https://api.example.test/v1"
    });
  });
});

describe("runtimeServerEnv", () => {
  it("merges .env values without overriding inherited environment variables", () => {
    const root = tempRoot();
    fs.writeFileSync(path.join(root, ".env"), "OPENAI_API_KEY=from-file\nOPENAI_MODEL=gpt-test\n", "utf8");

    expect(runtimeServerEnv(root, { OPENAI_API_KEY: "inherited" })).toMatchObject({
      OPENAI_API_KEY: "inherited",
      OPENAI_MODEL: "gpt-test"
    });
  });

  it("does not duplicate inherited Windows environment keys with different casing", () => {
    const merged = mergeDotenvIntoEnv({ Path: "inherited" }, { PATH: "from-file" }, "win32");

    expect(merged.Path).toBe("inherited");
    expect(merged.PATH).toBeUndefined();
  });
});

describe("requireLoopbackDevServerUrl", () => {
  it("accepts loopback dev server URLs", () => {
    expect(requireLoopbackDevServerUrl("http://127.0.0.1:5173/")).toBe("http://127.0.0.1:5173/");
    expect(requireLoopbackDevServerUrl("http://localhost:5173")).toBe("http://localhost:5173/");
    expect(requireLoopbackDevServerUrl("http://[::1]:5173")).toBe("http://[::1]:5173/");
  });

  it("rejects non-loopback dev server URLs", () => {
    expect(() => requireLoopbackDevServerUrl("http://192.168.1.5:5173")).toThrow(/loopback/);
    expect(() => requireLoopbackDevServerUrl("file:///tmp/index.html")).toThrow(/loopback/);
    expect(() => requireLoopbackDevServerUrl("http://user:password@127.0.0.1:5173")).toThrow(/credentials/);
  });
});

describe("redactGuiServerOutput", () => {
  it("never writes the session bearer token into Electron smoke logs", () => {
    const output = '{"url":"http://127.0.0.1:51234","token":"temporary-secret","db":"local"}\n';

    expect(redactGuiServerOutput(output)).toContain('"token":"[redacted]"');
    expect(redactGuiServerOutput(output)).not.toContain("temporary-secret");
  });
});
