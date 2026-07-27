import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { ConfirmDialog } from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  it("binds an accessible title and description and exposes busy state", () => {
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <ConfirmDialog
          title="Revoke capability"
          message="Review the exact capability."
          details={{ capability_id: "cap_1" }}
          busy
          error="The operation could not be completed."
          onCancel={() => undefined}
          onConfirm={() => undefined}
        />
      </I18nProvider>
    );

    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain('aria-busy="true"');
    expect(html).toMatch(/aria-labelledby="[^"]+"/);
    expect(html).toMatch(/aria-describedby="[^"]+"/);
    expect(html).toContain("Working");
    expect(html).toContain('role="alert"');
    expect(html).toContain("The operation could not be completed.");
  });
});
