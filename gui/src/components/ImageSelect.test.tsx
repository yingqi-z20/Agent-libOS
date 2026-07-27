import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { ImageSelect } from "./ImageSelect";

describe("ImageSelect", () => {
  it("gives the preset and manual image controls distinct accessible names", () => {
    const html = renderToStaticMarkup(
      <I18nProvider initialLanguage="en">
        <ImageSelect
          images={[{
            image_id: "coding-agent:v0",
            name: "coding-agent",
            version: "v0",
            boot_kind: "fresh",
            default_tools: [],
            default_skills: [],
            required_capabilities_count: 0,
            required_modules_count: 0
          }]}
          value="coding-agent:v0"
          label="Agent image"
          onChange={() => undefined}
        />
      </I18nProvider>
    );

    expect(html).toContain('role="group"');
    expect(html).toMatch(/<select aria-labelledby="[^"]+"/);
    expect(html).toContain('aria-label="Agent image: Image id"');
  });
});
