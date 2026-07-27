// @vitest-environment jsdom

import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ImageSummary, LLMProfileInput, LLMProfileSummary } from "../api/types";
import { I18nProvider } from "../i18n";
import {
  UserTaskSettingsDialog,
  type TaskLaunchSettings
} from "./UserTaskSettingsDialog";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const roots: Root[] = [];

afterEach(async () => {
  for (const root of roots.splice(0)) {
    await act(() => root.unmount());
  }
  document.body.replaceChildren();
});

describe("UserTaskSettingsDialog", () => {
  it("renders all eight launch settings and applies the complete draft atomically", async () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    const container = await renderDialog({ onApply, onClose });

    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    expect(container.querySelector(".taskSettingsSection:nth-of-type(1) .imageSelect")).not.toBeNull();
    expect(container.querySelector(".taskSettingsSection:nth-of-type(1) .llmProfileSelect")).not.toBeNull();
    expect(field(container, "Quanta")).not.toBeNull();
    expect(field(container, "Initial working directory")).not.toBeNull();
    expect(field(container, "Workspace permission requests")).not.toBeNull();
    expect(field(container, "Allow requests for local Git operations")).not.toBeNull();
    expect(field(container, "Command execution")).not.toBeNull();
    expect(field(container, "Enable persistent context enrichment and bounded maintenance")).not.toBeNull();

    await setInput(container.querySelector<HTMLInputElement>(".imageSelect input"), "research-agent:v2");
    await setSelect(container.querySelector<HTMLSelectElement>(".llmProfileSelect select"), "beta");
    await setInput(inputFor(container, "Quanta"), "24");
    await setInput(inputFor(container, "Initial working directory"), "packages/gui/../web");
    await setSelect(selectFor(container, "Workspace permission requests"), "manage");
    await click(checkboxFor(container, "Allow requests for local Git operations"));
    await setSelect(selectFor(container, "Command execution"), "reviewed");
    await click(checkboxFor(container, "Enable persistent context enrichment and bounded maintenance"));
    await click(button(container, "Save settings"));

    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply).toHaveBeenCalledWith({
      image: "research-agent:v2",
      llmProfile: "beta",
      maxQuantaInput: "24",
      workingDirectory: "packages/gui/../web",
      workspaceAccess: "manage",
      allowGitRequests: false,
      commandAccess: "reviewed",
      contextMaintenance: true
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("discards local edits on Cancel, Escape, and backdrop close", async () => {
    const onApply = vi.fn();
    const onClose = vi.fn();
    const container = await renderHarness({ onApply, onClose });

    await setInput(inputFor(container, "Quanta"), "9");
    await click(button(container, "Cancel"));
    await reopen(container);
    expect(inputFor(container, "Quanta").value).toBe("12");

    await setInput(inputFor(container, "Quanta"), "18");
    await act(() => {
      container.querySelector<HTMLElement>('[role="dialog"]')?.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Escape", bubbles: true })
      );
    });
    await reopen(container);
    expect(inputFor(container, "Quanta").value).toBe("12");

    await setInput(inputFor(container, "Quanta"), "27");
    await act(() => {
      container.querySelector<HTMLElement>(".modalBackdrop")?.dispatchEvent(
        new MouseEvent("mousedown", { bubbles: true })
      );
    });
    await reopen(container);
    expect(inputFor(container, "Quanta").value).toBe("12");

    expect(onApply).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it("shows inline validation and blocks save for invalid quanta or working directories", async () => {
    const onApply = vi.fn();
    const container = await renderDialog({ onApply });
    const save = button(container, "Save settings");

    await setInput(inputFor(container, "Quanta"), "1.5");
    expect(inputFor(container, "Quanta").getAttribute("aria-invalid")).toBe("true");
    expect(container.textContent).toContain("Maximum quanta must be a positive whole number");
    expect(save.disabled).toBe(true);

    await setInput(inputFor(container, "Quanta"), "20");
    await setInput(inputFor(container, "Initial working directory"), "/tmp/outside");
    expect(inputFor(container, "Initial working directory").getAttribute("aria-invalid")).toBe("true");
    expect(container.textContent).toContain("Use a relative path inside the workspace");
    expect(save.disabled).toBe(true);

    await setInput(inputFor(container, "Initial working directory"), "../outside");
    expect(save.disabled).toBe(true);
    await click(save);
    expect(onApply).not.toHaveBeenCalled();
  });

  it("keeps one dialog and preserves the draft while managing profiles", async () => {
    const container = await renderDialog();
    await setInput(inputFor(container, "Quanta"), "37");

    await click(button(container, "Manage"));
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    expect(container.textContent).toContain("Model profiles");

    await click(container.querySelector<HTMLButtonElement>(".modalActions .secondary"));
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    expect(inputFor(container, "Quanta").value).toBe("37");
  });

  it("restores focus to the settings launcher after a profile-manager round trip", async () => {
    const container = await renderHarness({ initialOpen: false });
    const launcher = button(container, "Open settings");
    launcher.focus();

    await click(launcher);
    expect(document.activeElement).toBe(container.querySelector('[role="dialog"]'));
    await click(button(container, "Manage"));
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    await click(container.querySelector<HTMLButtonElement>(".modalActions .secondary"));
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    await click(button(container, "Cancel"));

    expect(document.activeElement).toBe(launcher);
  });

  it("falls back to the runtime default after deleting the selected profile", async () => {
    const onDelete = vi.fn(async () => true);
    const container = await renderDialog({ onDelete });

    await click(button(container, "Manage"));
    await click(container.querySelector<HTMLButtonElement>('[aria-label="Delete profile: alpha"]'));
    await click(container.querySelector<HTMLButtonElement>(".inlineConfirm button.danger"));
    expect(onDelete).toHaveBeenCalledWith("alpha");

    await click(container.querySelector<HTMLButtonElement>(".modalActions .secondary"));
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    expect(container.querySelector<HTMLSelectElement>(".llmProfileSelect select")?.value).toBe("");
  });

  it("disables every editable control while the surrounding task flow is busy", async () => {
    const container = await renderDialog({ busy: true });

    const controls = Array.from(container.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLButtonElement>(
      '.taskSettingsModal input, .taskSettingsModal select, .taskSettingsModal button'
    ));
    expect(controls.length).toBeGreaterThan(8);
    expect(controls.every((control) => control.disabled)).toBe(true);
  });

  it("renders the complete Chinese task-settings vocabulary", async () => {
    const container = await renderDialog({ language: "zh-CN" });

    expect(container.textContent).toContain("新任务设置");
    expect(container.textContent).toContain("运行时");
    expect(container.textContent).toContain("权限");
    expect(container.textContent).toContain("保存设置");
    expect(container.textContent).toContain("启用持久上下文增强及受限维护");
  });
});

type RenderOptions = {
  busy?: boolean;
  initialOpen?: boolean;
  language?: "en" | "zh-CN";
  onApply?: (next: TaskLaunchSettings) => void;
  onClose?: () => void;
  onDelete?: (profileId: string) => Promise<boolean>;
};

async function renderDialog(options: RenderOptions = {}): Promise<HTMLDivElement> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  roots.push(root);
  await act(() => {
    root.render(
      <I18nProvider initialLanguage={options.language ?? "en"}>
        <UserTaskSettingsDialog
          value={initialSettings}
          images={images}
          llmProfiles={[profile("alpha"), profile("beta")]}
          busy={options.busy}
          onApply={options.onApply ?? (() => undefined)}
          onClose={options.onClose ?? (() => undefined)}
          onCreateLlmProfile={async () => true}
          onUpdateLlmProfile={async () => true}
          onDeleteLlmProfile={options.onDelete ?? (async () => true)}
        />
      </I18nProvider>
    );
  });
  return container;
}

async function renderHarness(options: RenderOptions = {}): Promise<HTMLDivElement> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  roots.push(root);
  await act(() => {
    root.render(
      <I18nProvider initialLanguage={options.language ?? "en"}>
        <DialogHarness {...options} />
      </I18nProvider>
    );
  });
  return container;
}

function DialogHarness({ busy, initialOpen = true, onApply, onClose, onDelete }: RenderOptions) {
  const [open, setOpen] = useState(initialOpen);
  const [settings, setSettings] = useState(initialSettings);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open settings</button>
      {open ? (
        <UserTaskSettingsDialog
          value={settings}
          images={images}
          llmProfiles={[profile("alpha"), profile("beta")]}
          busy={busy}
          onApply={(next) => {
            setSettings(next);
            onApply?.(next);
          }}
          onClose={() => {
            setOpen(false);
            onClose?.();
          }}
          onCreateLlmProfile={async (_profile: LLMProfileInput) => true}
          onUpdateLlmProfile={async () => true}
          onDeleteLlmProfile={onDelete ?? (async () => true)}
        />
      ) : null}
    </>
  );
}

const initialSettings: TaskLaunchSettings = {
  image: "coding-agent:v0",
  llmProfile: "alpha",
  maxQuantaInput: "12",
  workingDirectory: "gui",
  workspaceAccess: "edit",
  allowGitRequests: true,
  commandAccess: "none",
  contextMaintenance: false
};

const images: ImageSummary[] = [
  {
    image_id: "coding-agent:v0",
    name: "coding-agent",
    version: "v0",
    boot_kind: "llm",
    default_tools: [],
    default_skills: [],
    required_capabilities_count: 0,
    required_modules_count: 0
  },
  {
    image_id: "research-agent:v2",
    name: "research-agent",
    version: "v2",
    boot_kind: "llm",
    default_tools: [],
    default_skills: [],
    required_capabilities_count: 0,
    required_modules_count: 0
  }
];

function profile(profileId: string): LLMProfileSummary {
  return {
    profile_id: profileId,
    model: `model-${profileId}`,
    base_url: null,
    api_key_env: "OPENAI_API_KEY",
    api_key_env_present: true,
    api_mode: "responses",
    timeout_s: null,
    max_retries: null,
    store: null,
    reasoning_effort: null,
    verbosity: null,
    safety_identifier_env: null,
    prompt_cache_retention: null,
    responses_previous_response_id: null,
    parallel_tool_calls: null,
    auto_wait_on_empty_tool_calls: null,
    fallback_json_actions: null,
    temperature: null,
    max_tokens: null,
    context_window_tokens: null,
    allow_custom_base_url: false,
    source: "user",
    editable: true,
    is_default: false
  };
}

function field(container: ParentNode, text: string): HTMLLabelElement | null {
  return Array.from(container.querySelectorAll<HTMLLabelElement>("label"))
    .find((label) => label.textContent?.includes(text)) ?? null;
}

function inputFor(container: ParentNode, text: string): HTMLInputElement {
  const input = field(container, text)?.querySelector<HTMLInputElement>("input");
  if (!input) throw new Error(`Missing input for ${text}`);
  return input;
}

function selectFor(container: ParentNode, text: string): HTMLSelectElement {
  const select = field(container, text)?.querySelector<HTMLSelectElement>("select");
  if (!select) throw new Error(`Missing select for ${text}`);
  return select;
}

function checkboxFor(container: ParentNode, text: string): HTMLInputElement {
  const input = inputFor(container, text);
  if (input.type !== "checkbox") throw new Error(`Expected checkbox for ${text}`);
  return input;
}

function button(container: ParentNode, text: string): HTMLButtonElement {
  const selected = Array.from(container.querySelectorAll<HTMLButtonElement>("button"))
    .find((candidate) => candidate.textContent?.trim() === text);
  if (!selected) throw new Error(`Missing button ${text}`);
  return selected;
}

async function setInput(input: HTMLInputElement | null, value: string) {
  if (!input) throw new Error("Missing input");
  await act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function setSelect(select: HTMLSelectElement | null, value: string) {
  if (!select) throw new Error("Missing select");
  await act(() => {
    Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set?.call(select, value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function click(element: HTMLElement | null) {
  if (!element) throw new Error("Missing clickable element");
  await act(async () => {
    element.click();
    await Promise.resolve();
  });
}

async function reopen(container: ParentNode) {
  await click(button(container, "Open settings"));
}
