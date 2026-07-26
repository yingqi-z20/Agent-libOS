import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { SchedulerStatus } from "../api/types";
import { I18nProvider } from "../i18n";
import { TopBar } from "./TopBar";

describe("TopBar", () => {
  it("associates the quanta error and names compact operator actions", () => {
    const html = renderTopBar({ quantaValid: false, pendingHumanCount: 2 });
    const quanta = html.match(/<input[^>]*aria-invalid="true"[^>]*>/)?.[0] ?? "";
    const errorId = quanta.match(/aria-errormessage="([^"]+)"/)?.[1];

    expect(errorId).toBeTruthy();
    expect(html).toContain(`id="${errorId}"`);
    expect(html).toContain('aria-label="Open SQLite database"');
    expect(html).toContain('aria-label="Refresh snapshot"');
    expect(html).toContain('aria-label="Pending human requests: 2"');
    expect(html).toContain('aria-label="User Page"');
  });

  it("shows one explicit busy status in place of the scheduler pill", () => {
    const html = renderTopBar({ busy: true });

    expect(html).toContain('class="operatorBusyStatus"');
    expect(html).toContain("Working…");
    expect(html).not.toContain('class="schedulerPill');
  });

  it("disables spawn, run, and step for an invalid quanta draft", () => {
    const invalid = renderTopBar({ quantaValid: false, selectedPid: "pid_selected" });
    const valid = renderTopBar({ quantaValid: true, selectedPid: "pid_selected" });
    const tag = (html: string, marker: string) => html.match(new RegExp(`<button[^>]*${marker}[^>]*>`))?.[0] ?? "";

    expect(tag(invalid, 'class="primary spawnProcessButton"')).toContain("disabled");
    expect(tag(invalid, 'title="Run selected process"')).toContain("disabled");
    expect(tag(invalid, 'title="Step selected process"')).toContain("disabled");
    expect(tag(valid, 'class="primary spawnProcessButton"')).not.toContain("disabled");
    expect(tag(valid, 'title="Run selected process"')).not.toContain("disabled");
    expect(tag(valid, 'title="Step selected process"')).not.toContain("disabled");
  });
});

function renderTopBar(overrides: Partial<React.ComponentProps<typeof TopBar>> = {}): string {
  return renderToStaticMarkup(
    <I18nProvider initialLanguage="en">
      <TopBar
        db="/tmp/operator.sqlite"
        scheduler={scheduler()}
        maxQuantaInput=""
        selectedPid={null}
        onMaxQuantaChange={() => undefined}
        onOpenDb={() => undefined}
        onSpawn={() => undefined}
        onRun={() => undefined}
        onStep={() => undefined}
        onPause={() => undefined}
        onAutoRunChange={() => undefined}
        onRefresh={() => undefined}
        onOpenPending={() => undefined}
        onShowUser={() => undefined}
        busy={false}
        streamStatus="connected"
        lastUpdatedAt={null}
        {...overrides}
      />
    </I18nProvider>
  );
}

function scheduler(): SchedulerStatus {
  return {
    auto_run: false,
    running: false,
    paused: true,
    task_id: null,
    reason: null,
    last_result: [],
    last_error: null,
    started_at: null,
    finished_at: null,
    default_max_quanta: null
  };
}
