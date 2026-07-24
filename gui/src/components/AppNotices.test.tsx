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
          onDismissError={() => undefined}
          onRetry={() => undefined}
        />
      </I18nProvider>
    );

    expect(markup).toContain('class="appNotice warning snapshotWarning"');
    expect(markup).toContain("… (+1)");
    expect(markup).toContain('aria-label="Dismiss"');
  });
});
