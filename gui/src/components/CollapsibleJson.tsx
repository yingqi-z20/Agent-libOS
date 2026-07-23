import { ChevronDown, ChevronRight } from "lucide-react";
import { useId, useMemo, useState } from "react";
import { useI18n, type TranslationKey } from "../i18n";

type CollapsibleJsonProps = {
  value: unknown;
  label?: string;
  defaultExpanded?: boolean;
};

export function CollapsibleJson({ value, label, defaultExpanded = false }: CollapsibleJsonProps) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState(defaultExpanded);
  const contentId = useId();
  const pretty = useMemo(() => JSON.stringify(value, null, 2), [value]);
  const preview = useMemo(() => compactPreview(value, t), [value, t]);
  const Icon = expanded ? ChevronDown : ChevronRight;
  const resolvedLabel = label ?? t("json.details");

  return (
    <div className="collapsibleJson">
      <button
        className="collapseToggle"
        type="button"
        aria-expanded={expanded}
        aria-controls={contentId}
        onClick={() => setExpanded((current) => !current)}
      >
        <Icon size={14} />
        <span>{expanded ? t("json.hide") : t("json.show")} {resolvedLabel}</span>
        <span className="collapseMeta">{metadata(value, pretty, t)}</span>
      </button>
      {!expanded ? <div id={contentId} className="jsonPreview" title={preview}>{preview}</div> : null}
      {expanded ? <pre id={contentId} className="jsonBlock" tabIndex={0}>{pretty}</pre> : null}
    </div>
  );
}

function compactPreview(value: unknown, t: Translate): string {
  if (value === null) return t("json.null");
  if (value === undefined) return t("json.undefined");
  if (typeof value === "string") return value || t("json.empty");
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    if (value.length === 0) return t("json.arrayEmpty");
    return t("json.arrayPreview", {
      count: value.length,
      itemLabel: t(value.length === 1 ? "json.item" : "json.items"),
      preview: JSON.stringify(value.slice(0, 2))
    });
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return t("json.objectEmpty");
    const first = entries.slice(0, 4).map(([key, item]) => `${key}: ${shortValue(item)}`).join(", ");
    return `{${first}${entries.length > 4 ? ", ..." : ""}}`;
  }
  return String(value);
}

function shortValue(value: unknown): string {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (typeof value === "string") return JSON.stringify(value.length > 60 ? `${value.slice(0, 57)}...` : value);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `[${value.length}]`;
  if (typeof value === "object") return `{${Object.keys(value as Record<string, unknown>).length}}`;
  return String(value);
}

function metadata(value: unknown, pretty: string, t: Translate): string {
  const bytes = new Blob([pretty]).size;
  const size = bytes < 1024 ? `${bytes} B` : `${Math.ceil(bytes / 1024)} KB`;
  if (Array.isArray(value)) return t("json.itemsMeta", { count: value.length, size });
  if (value && typeof value === "object") return t("json.fields", { count: Object.keys(value as Record<string, unknown>).length, size });
  return size;
}

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;
