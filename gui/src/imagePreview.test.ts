import { describe, expect, it } from "vitest";
import { previewImageManifest } from "./imagePreview";

describe("previewImageManifest", () => {
  it("previews JSON image manifests", () => {
    const preview = previewImageManifest(JSON.stringify({
      image: {
        image_id: "json-agent:v0",
        name: "json-agent",
        version: "v0",
        default_tools: ["echo"],
        required_capabilities: [{ resource: "filesystem:/README.md", rights: ["read"] }],
        required_modules: [{ module_id: "module:v0", source_sha256: "0".repeat(64) }]
      }
    }));

    expect(preview).toMatchObject({
      image_id: "json-agent:v0",
      name: "json-agent",
      version: "v0",
      default_tools_count: 1,
      required_capabilities_count: 1,
      required_modules_count: 1
    });
  });

  it("does not approximate IMAGE.yaml with a parser that differs from the backend", () => {
    const preview = previewImageManifest(`
image:
  image_id: package-agent:v0
  name: package-agent
  version: v1
  default_tools:
    - echo
    - list_files
  required_capabilities:
    - resource: filesystem:/README.md
      rights: [read]
  required_modules:
    - module_id: module:v0
      source_sha256: ${"0".repeat(64)}
`);

    expect(preview).toMatchObject({
      image_id: null,
      name: null,
      version: null,
      default_tools_count: null,
      required_capabilities_count: null,
      required_modules_count: null
    });
  });
});
