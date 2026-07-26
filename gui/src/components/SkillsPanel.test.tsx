// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import type { LibOSClient } from "../api/client";
import type { RuntimeProcess, SkillSummary } from "../api/types";
import type { ConfirmationRequest } from "../adminTypes";
import { I18nProvider } from "../i18n";
import { packageSha256, SkillsPanel } from "./SkillsPanel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("SkillsPanel activation identity", () => {
  it("reactivates a replaced catalog package and freezes its discovered hash", async () => {
    const loadedHash = "a".repeat(64);
    const discoveredHash = "b".repeat(64);
    const laterHash = "c".repeat(64);
    const activateSkill = vi.fn().mockResolvedValue({});
    const client = {
      activateSkill,
      inspectSkill: vi.fn().mockResolvedValue({}),
      unloadSkill: vi.fn().mockResolvedValue({})
    } as unknown as LibOSClient;
    const confirmations: ConfirmationRequest[] = [];
    const container = document.createElement("div");
    const root = createRoot(container);

    await act(() => {
      root.render(panel({
        client,
        process: processWithLoadedHash(loadedHash),
        skills: [skill(discoveredHash)],
        confirmAction: (request) => { confirmations.push(request); }
      }));
    });

    const activate = container.querySelector<HTMLButtonElement>(".adminActions button.warning");
    const unload = container.querySelector<HTMLButtonElement>(".adminActions button.danger");
    expect(activate?.disabled).toBe(false);
    expect(unload?.disabled).toBe(false);
    await act(() => activate?.click());
    const confirmation = confirmations[0];
    expect(confirmation?.details).toMatchObject({
      skill_id: "reviewer",
      expected_package_sha256: discoveredHash
    });

    await act(() => {
      root.render(panel({
        client,
        process: processWithLoadedHash(loadedHash),
        skills: [skill(laterHash)],
        confirmAction: () => undefined
      }));
    });
    await act(async () => {
      await confirmation?.action();
    });

    expect(activateSkill).toHaveBeenCalledWith(
      "reviewer",
      "pid_1",
      discoveredHash,
      true,
      "pid_1"
    );
    await act(() => root.unmount());
  });

  it("disables activation for the exact loaded package or a missing catalog hash", async () => {
    const currentHash = "d".repeat(64);
    const client = {
      activateSkill: vi.fn(),
      inspectSkill: vi.fn(),
      unloadSkill: vi.fn()
    } as unknown as LibOSClient;
    const container = document.createElement("div");
    const root = createRoot(container);

    await act(() => {
      root.render(panel({
        client,
        process: processWithLoadedHash(currentHash),
        skills: [skill(currentHash)],
        confirmAction: () => undefined
      }));
    });
    expect(container.querySelector<HTMLButtonElement>(".adminActions button.warning")?.disabled).toBe(true);

    await act(() => {
      root.render(panel({
        client,
        process: processWithLoadedHash(currentHash),
        skills: [{ skill_id: "reviewer", name: "Reviewer" }],
        confirmAction: () => undefined
      }));
    });
    expect(container.querySelector<HTMLButtonElement>(".adminActions button.warning")?.disabled).toBe(true);
    await act(() => root.unmount());
  });

  it("accepts only canonical lowercase SHA-256 package identities", () => {
    expect(packageSha256("e".repeat(64))).toBe("e".repeat(64));
    expect(packageSha256("E".repeat(64))).toBeNull();
    expect(packageSha256("e".repeat(63))).toBeNull();
    expect(packageSha256(undefined)).toBeNull();
  });
});

function panel({
  client,
  process,
  skills,
  confirmAction
}: {
  client: LibOSClient;
  process: RuntimeProcess;
  skills: SkillSummary[];
  confirmAction(request: ConfirmationRequest): void;
}) {
  return (
    <I18nProvider initialLanguage="en">
      <SkillsPanel
        process={process}
        skills={skills}
        tools={[]}
        client={client}
        confirmAction={confirmAction}
      />
    </I18nProvider>
  );
}

function skill(packageSha256Value: string): SkillSummary {
  return {
    skill_id: "reviewer",
    name: "Reviewer",
    package_sha256: packageSha256Value
  };
}

function processWithLoadedHash(packageSha256Value: string): RuntimeProcess {
  return {
    pid: "pid_1",
    parent_pid: null,
    image_id: "base-agent:v0",
    llm_profile_id: "default",
    status: "waiting",
    goal_oid: null,
    checkpoint_head: null,
    working_directory: ".",
    status_message: null,
    wait_state: null,
    outcome: null,
    state_generation: 1,
    loaded_skills: {
      reviewer: { package_sha256: packageSha256Value }
    },
    tool_table: {},
    capabilities: [],
    terminal: false,
    unread_message_count: 0,
    interrupt_count: 0,
    messages: [],
    llm_call_count: 0,
    token_total: 0,
    rating: null
  };
}
