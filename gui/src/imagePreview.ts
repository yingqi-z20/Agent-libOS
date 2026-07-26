export type ImageManifestPreview = {
  image_id: string | null;
  name: string | null;
  version: string | null;
  default_tools_count: number | null;
  required_capabilities_count: number | null;
  required_modules_count: number | null;
  bytes: number;
};

export function previewImageManifest(text: string): ImageManifestPreview {
  const bytes = new Blob([text]).size;
  const parsed = parseJsonPreview(text);
  if (parsed) return { ...parsed, bytes };
  return {
    // YAML is intentionally not approximated with regular expressions. The
    // backend's bounded, unique-key YAML loader is the authority; confirmation
    // binds the exact manifest SHA-256 supplied by Electron instead.
    image_id: null,
    name: null,
    version: null,
    default_tools_count: null,
    required_capabilities_count: null,
    required_modules_count: null,
    bytes
  };
}

function parseJsonPreview(text: string): Omit<ImageManifestPreview, "bytes"> | null {
  try {
    const value = JSON.parse(text) as unknown;
    const image = unwrapImage(value);
    if (!image) return null;
    return {
      image_id: stringValue(image.image_id),
      name: stringValue(image.name),
      version: stringValue(image.version),
      default_tools_count: Array.isArray(image.default_tools) ? image.default_tools.length : null,
      required_capabilities_count: Array.isArray(image.required_capabilities) ? image.required_capabilities.length : null,
      required_modules_count: Array.isArray(image.required_modules) ? image.required_modules.length : null
    };
  } catch {
    return null;
  }
}

function unwrapImage(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const image = record.image;
  if (image && typeof image === "object" && !Array.isArray(image)) return image as Record<string, unknown>;
  return record;
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}
