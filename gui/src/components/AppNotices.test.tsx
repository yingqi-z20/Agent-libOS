import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RuntimeSnapshot } from "../api/types";
import { I18nProvider } from "../i18n";
import { AppNotices, summarizeTruncatedSections } from "./AppNotices";

describe("summarizeTruncatedSections", () => {
  it("bounds long snapshot section lists without hiding the total remainder", () => {
    expect(summarizeTruncatedSections(["one", "two", "three", "four", "five"])).toBe(
      "one, two, three … (+2)"
    );
  });

  it("keeps short lists exact", () => {
    expect(summarizeTruncatedSections(["one", "two"])).toBe("one, two");
  });

  it("renders snapshot truncation as a bounded dismissible notice", () => {
    const snapshot = {
      scheduler: { last_error: null },
      _truncated: { one: {}, two: {}, three: {}, four: {} }
    } as unknown as RuntimeSnapshot;

    const markup = renderToStaticMarkup(
      <I18nProvider>
        <AppNotices
          error={null}
          snapshot={snapshot}
          streamStatus="connected"
          refreshing={false}
          showSnapshotDiagnostics
          onDismissError={() => undefined}
          onRetry={() => undefined}
        />
      </I18nProvider>
    );

    expect(markup).toContain('class="appNotice warning snapshotWarning"');
    expect(markup).toContain('data-has-visible="true"');
    expect(markup).toContain("… (+1)");
    expect(markup).toContain('aria-label="Dismiss"');
  });

  it("keeps snapshot diagnostics out of the standard user surface", () => {
    const snapshot = {
      scheduler: { last_error: null },
      _truncated: { events: {} }
    } as unknown as RuntimeSnapshot;

    const markup = renderToStaticMarkup(
      <I18nProvider>
        <AppNotices
          error={null}
          snapshot={snapshot}
          streamStatus="connected"
          refreshing={false}
          onDismissError={() => undefined}
          onRetry={() => undefined}
        />
      </I18nProvider>
    );

    expect(markup).not.toContain("snapshotWarning");
    expect(markup).not.toContain("sections were shortened");
    expect(markup).not.toContain("data-has-visible");
  });

  it("announces pending Human requests from every process", () => {
    const snapshot = {
      scheduler: { last_error: null },
      human_requests: [
        { request_id: "req-1", pid: "pid-1", status: "pending" },
        { request_id: "req-2", pid: "pid-2", status: "pending" },
        { request_id: "req-3", pid: "pid-3", status: "responded" }
      ]
    } as unknown as RuntimeSnapshot;

    const markup = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <AppNotices
          error={null}
          snapshot={snapshot}
          streamStatus="connected"
          refreshing={false}
          onDismissError={() => undefined}
          onRetry={() => undefined}
        />
      </I18nProvider>
    );

    expect(markup).toContain("Pending human requests: 2");
    expect(markup).toContain('role="status"');
    expect(markup).not.toContain("data-has-visible");
  });
});
