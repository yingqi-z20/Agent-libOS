import type { McpInputRequest } from "../api/types";

const MAX_INPUT_REQUESTS = 16;
const MAX_SCHEMA_PROPERTIES = 64;
const MAX_OPTIONS = 256;
const MAX_TEXT_CHARS = 8_192;
const MAX_RESPONSE_BYTES = 256 * 1_024;

export type ElicitationAction = "" | "accept" | "decline" | "cancel";

export type ElicitationFieldDraft = {
  included: boolean;
  raw: string;
  selected: string[];
};

export type ElicitationRequestDraft = {
  action: ElicitationAction;
  urlReviewed: boolean;
  fields: Record<string, ElicitationFieldDraft>;
};

export type ElicitationDrafts = Record<string, ElicitationRequestDraft>;

export type ElicitationOption = { value: string; label: string };

export type ElicitationProperty = {
  name: string;
  label: string;
  description: string | null;
  required: boolean;
  kind: "string" | "number" | "integer" | "boolean" | "array";
  format: "email" | "uri" | "date" | "date-time" | null;
  options: ElicitationOption[];
  minimum: number | null;
  maximum: number | null;
  minimumSize: number | null;
  maximumSize: number | null;
  hasDefault: boolean;
  defaultValue: unknown;
};

export type ElicitationPlan = {
  requestId: string;
  prompt: string;
  mode: "form" | "url";
  inertUrl: string | null;
  properties: ElicitationProperty[];
};

export type ElicitationInspection = {
  plans: ElicitationPlan[];
  error: string | null;
};

export function inspectElicitationRequests(requests: McpInputRequest[]): ElicitationInspection {
  try {
    return { plans: parseElicitationRequests(requests), error: null };
  } catch (selected) {
    return {
      plans: [],
      error: selected instanceof Error ? selected.message : String(selected)
    };
  }
}

export function initializeElicitationDrafts(requests: McpInputRequest[]): ElicitationDrafts {
  const inspection = inspectElicitationRequests(requests);
  const drafts = emptyRecord<ElicitationRequestDraft>();
  for (const plan of inspection.plans) {
    const fields = emptyRecord<ElicitationFieldDraft>();
    for (const property of plan.properties) {
      const selected = property.kind === "array" && Array.isArray(property.defaultValue)
        ? property.defaultValue.filter((item): item is string => typeof item === "string")
        : [];
      fields[property.name] = {
        included: property.required,
        raw: property.hasDefault && property.kind !== "array"
          ? primitiveDraftText(property.defaultValue)
          : "",
        selected
      };
    }
    drafts[plan.requestId] = {
      action: "",
      urlReviewed: false,
      fields
    };
  }
  return drafts;
}

export function elicitationResponsesReady(
  requests: McpInputRequest[],
  drafts: ElicitationDrafts
): boolean {
  try {
    buildElicitationResponses(requests, drafts);
    return true;
  } catch {
    return false;
  }
}

export function buildElicitationResponses(
  requests: McpInputRequest[],
  drafts: ElicitationDrafts
): Record<string, unknown> {
  const plans = parseElicitationRequests(requests);
  const expectedIds = new Set(plans.map((plan) => plan.requestId));
  if (Object.keys(drafts).some((requestId) => !expectedIds.has(requestId))) {
    throw new Error("MCP Elicitation answers no longer match the displayed requests.");
  }
  const responses = emptyRecord<unknown>();
  for (const plan of plans) {
    const draft = drafts[plan.requestId];
    if (!draft || !["accept", "decline", "cancel"].includes(draft.action)) {
      throw new Error(`Choose an explicit response action for ${plan.requestId}.`);
    }
    if (draft.action !== "accept") {
      responses[plan.requestId] = { action: draft.action };
      continue;
    }
    if (plan.mode === "url") {
      if (!draft.urlReviewed) {
        throw new Error(`Review the inert URL explicitly before accepting ${plan.requestId}.`);
      }
      responses[plan.requestId] = { action: "accept" };
      continue;
    }
    const content = emptyRecord<unknown>();
    const expectedFields = new Set(plan.properties.map((property) => property.name));
    if (Object.keys(draft.fields).some((name) => !expectedFields.has(name))) {
      throw new Error(`MCP Elicitation fields changed for ${plan.requestId}.`);
    }
    for (const property of plan.properties) {
      const field = draft.fields[property.name];
      if (!field) throw new Error(`MCP Elicitation field ${property.name} is missing.`);
      if (property.required && !field.included) {
        throw new Error(`Required MCP Elicitation field ${property.name} is missing.`);
      }
      if (!property.required && !field.included) continue;
      content[property.name] = fieldValue(property, field);
    }
    responses[plan.requestId] = { action: "accept", content };
  }
  const encoded = new TextEncoder().encode(JSON.stringify(responses));
  if (encoded.byteLength > MAX_RESPONSE_BYTES) {
    throw new Error("MCP Elicitation answers exceed the bounded response size.");
  }
  return responses;
}

function parseElicitationRequests(requests: McpInputRequest[]): ElicitationPlan[] {
  if (!Array.isArray(requests) || requests.length > MAX_INPUT_REQUESTS) {
    throw new Error("MCP Elicitation request count is unsupported.");
  }
  if (jsonByteLength(requests) > MAX_RESPONSE_BYTES) {
    throw new Error("MCP Elicitation requests exceed the bounded schema size.");
  }
  const ids = new Set<string>();
  return requests.map((request) => {
    if (!request.request_id || request.request_id.length > 256 || ids.has(request.request_id)) {
      throw new Error("MCP Elicitation request identity is malformed.");
    }
    ids.add(request.request_id);
    if (request.kind !== "elicitation") {
      throw new Error("Sampling and Roots requests are unsupported and cannot be answered.");
    }
    if (typeof request.prompt !== "string" || !request.prompt || request.prompt.length > MAX_TEXT_CHARS) {
      throw new Error(`MCP Elicitation prompt is malformed for ${request.request_id}.`);
    }
    if (request.mode === "url") {
      if (!isRecord(request.schema)) {
        throw new Error(`MCP URL Elicitation schema is malformed for ${request.request_id}.`);
      }
      const inertUrl = request.inert_url;
      if (typeof inertUrl !== "string" || !validInertUrl(inertUrl)) {
        throw new Error(`MCP URL Elicitation URL is malformed for ${request.request_id}.`);
      }
      return {
        requestId: request.request_id,
        prompt: request.prompt,
        mode: "url",
        inertUrl,
        properties: []
      };
    }
    if (request.mode !== "form") {
      throw new Error(`MCP Elicitation mode is unsupported for ${request.request_id}.`);
    }
    return {
      requestId: request.request_id,
      prompt: request.prompt,
      mode: "form",
      inertUrl: null,
      properties: parseObjectSchema(request.schema, request.request_id)
    };
  });
}

function parseObjectSchema(schema: unknown, requestId: string): ElicitationProperty[] {
  if (!isRecord(schema)
      || hasUnknownKeys(schema, new Set(["$schema", "type", "properties", "required"]))
      || schema.type !== "object"
      || !isRecord(schema.properties)) {
    throw new Error(`MCP Elicitation object schema is unsupported for ${requestId}.`);
  }
  if ("$schema" in schema
      && (typeof schema.$schema !== "string" || schema.$schema.length > 512)) {
    throw new Error(`MCP Elicitation schema identifier is malformed for ${requestId}.`);
  }
  const propertyEntries = Object.entries(schema.properties);
  if (propertyEntries.length > MAX_SCHEMA_PROPERTIES) {
    throw new Error(`MCP Elicitation schema has too many fields for ${requestId}.`);
  }
  const required = schema.required ?? [];
  if (!Array.isArray(required)
      || required.some((name) => typeof name !== "string")
      || new Set(required).size !== required.length) {
    throw new Error(`MCP Elicitation required fields are malformed for ${requestId}.`);
  }
  const names = new Set(propertyEntries.map(([name]) => name));
  if (required.some((name) => !names.has(String(name)))) {
    throw new Error(`MCP Elicitation required fields changed for ${requestId}.`);
  }
  return propertyEntries.map(([name, value]) => {
    if (!name || name.length > 128 || !isRecord(value)) {
      throw new Error(`MCP Elicitation field is malformed for ${requestId}.`);
    }
    return parseProperty(name, value, required.includes(name), requestId);
  });
}

function parseProperty(
  name: string,
  schema: Record<string, unknown>,
  required: boolean,
  requestId: string
): ElicitationProperty {
  const common = new Set(["type", "title", "description", "default"]);
  const kind = schema.type;
  if (kind !== "string" && kind !== "number" && kind !== "integer"
      && kind !== "boolean" && kind !== "array") {
    throw new Error(`MCP Elicitation field ${name} has an unsupported type.`);
  }
  const title = optionalBoundedText(schema.title, `${name} title`);
  const description = optionalBoundedText(schema.description, `${name} description`);
  let options: ElicitationOption[] = [];
  let format: ElicitationProperty["format"] = null;
  let minimum: number | null = null;
  let maximum: number | null = null;
  let minimumSize: number | null = null;
  let maximumSize: number | null = null;
  const allowed = new Set(common);

  if (kind === "string") {
    for (const key of ["minLength", "maxLength", "format", "enum", "enumNames", "oneOf"]) allowed.add(key);
    [minimumSize, maximumSize] = nonnegativeBounds(schema, "minLength", "maxLength", name);
    if (schema.format !== undefined) {
      if (schema.format !== "email" && schema.format !== "uri"
          && schema.format !== "date" && schema.format !== "date-time") {
        throw new Error(`MCP Elicitation field ${name} has an unsupported format.`);
      }
      format = schema.format;
    }
    if (schema.enum !== undefined && schema.oneOf !== undefined) {
      throw new Error(`MCP Elicitation field ${name} has ambiguous choices.`);
    }
    if (schema.enum !== undefined) {
      const values = stringOptions(schema.enum, name);
      const labels = schema.enumNames === undefined
        ? values
        : stringOptions(schema.enumNames, `${name} labels`);
      if (labels.length !== values.length) {
        throw new Error(`MCP Elicitation field ${name} choice labels are misaligned.`);
      }
      options = values.map((value, index) => ({ value, label: labels[index] ?? value }));
    } else if (schema.enumNames !== undefined) {
      throw new Error(`MCP Elicitation field ${name} has labels without choices.`);
    } else if (schema.oneOf !== undefined) {
      options = titledOptions(schema.oneOf, name);
    }
  } else if (kind === "number" || kind === "integer") {
    allowed.add("minimum");
    allowed.add("maximum");
    minimum = optionalFiniteNumber(schema.minimum, `${name} minimum`);
    maximum = optionalFiniteNumber(schema.maximum, `${name} maximum`);
    if (minimum !== null && maximum !== null && minimum > maximum) {
      throw new Error(`MCP Elicitation field ${name} has inconsistent numeric bounds.`);
    }
  } else if (kind === "array") {
    for (const key of ["items", "minItems", "maxItems"]) allowed.add(key);
    [minimumSize, maximumSize] = nonnegativeBounds(schema, "minItems", "maxItems", name);
    if (!isRecord(schema.items)) {
      throw new Error(`MCP Elicitation field ${name} has an unsupported multi-select.`);
    }
    if (hasExactKeys(schema.items, ["type", "enum"]) && schema.items.type === "string") {
      options = stringOptions(schema.items.enum, name).map((value) => ({ value, label: value }));
    } else if (hasExactKeys(schema.items, ["anyOf"])) {
      options = titledOptions(schema.items.anyOf, name);
    } else {
      throw new Error(`MCP Elicitation field ${name} has an unsupported multi-select.`);
    }
  }
  if (hasUnknownKeys(schema, allowed)) {
    throw new Error(`MCP Elicitation field ${name} contains unsupported schema keywords.`);
  }
  const property: ElicitationProperty = {
    name,
    label: title ?? name,
    description,
    required,
    kind,
    format,
    options,
    minimum,
    maximum,
    minimumSize,
    maximumSize,
    hasDefault: Object.prototype.hasOwnProperty.call(schema, "default"),
    defaultValue: schema.default
  };
  if (property.hasDefault) validatePropertyValue(property, property.defaultValue);
  if (requestId.length < 1) throw new Error("MCP Elicitation request identity is malformed.");
  return property;
}

function fieldValue(property: ElicitationProperty, draft: ElicitationFieldDraft): unknown {
  let selected: unknown;
  if (property.kind === "array") {
    selected = [...draft.selected];
  } else if (property.kind === "boolean") {
    if (draft.raw !== "true" && draft.raw !== "false") {
      throw new Error(`MCP Elicitation field ${property.name} needs a boolean value.`);
    }
    selected = draft.raw === "true";
  } else if (property.kind === "number" || property.kind === "integer") {
    if (!draft.raw.trim()) {
      throw new Error(`MCP Elicitation field ${property.name} needs a numeric value.`);
    }
    selected = Number(draft.raw);
  } else {
    selected = draft.raw;
  }
  validatePropertyValue(property, selected);
  return selected;
}

function validatePropertyValue(property: ElicitationProperty, value: unknown): void {
  if (property.kind === "string") {
    if (typeof value !== "string") throw new Error(`MCP Elicitation field ${property.name} must be text.`);
    const size = [...value].length;
    if (size > MAX_TEXT_CHARS
        || (property.minimumSize !== null && size < property.minimumSize)
        || (property.maximumSize !== null && size > property.maximumSize)) {
      throw new Error(`MCP Elicitation field ${property.name} violates its text bounds.`);
    }
    if (property.options.length && !property.options.some((option) => option.value === value)) {
      throw new Error(`MCP Elicitation field ${property.name} has an invalid choice.`);
    }
    validateStringFormat(property, value);
    return;
  }
  if (property.kind === "boolean") {
    if (typeof value !== "boolean") throw new Error(`MCP Elicitation field ${property.name} must be boolean.`);
    return;
  }
  if (property.kind === "number" || property.kind === "integer") {
    if (typeof value !== "number" || !Number.isFinite(value)
        || (property.kind === "integer" && !Number.isSafeInteger(value))) {
      throw new Error(`MCP Elicitation field ${property.name} has an invalid number.`);
    }
    if ((property.minimum !== null && value < property.minimum)
        || (property.maximum !== null && value > property.maximum)) {
      throw new Error(`MCP Elicitation field ${property.name} violates its numeric bounds.`);
    }
    return;
  }
  if (!Array.isArray(value)
      || value.some((item) => typeof item !== "string")
      || new Set(value).size !== value.length
      || value.some((item) => !property.options.some((option) => option.value === item))
      || (property.minimumSize !== null && value.length < property.minimumSize)
      || (property.maximumSize !== null && value.length > property.maximumSize)) {
    throw new Error(`MCP Elicitation field ${property.name} has an invalid multi-select value.`);
  }
}

function validateStringFormat(property: ElicitationProperty, value: string): void {
  if (property.format === "email" && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
    throw new Error(`MCP Elicitation field ${property.name} needs a valid email address.`);
  }
  if (property.format === "uri") {
    try {
      const selected = new URL(value);
      if (!selected.protocol) throw new Error("missing scheme");
    } catch {
      throw new Error(`MCP Elicitation field ${property.name} needs an absolute URI.`);
    }
  }
  if (property.format === "date" && !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(`MCP Elicitation field ${property.name} needs an ISO date.`);
  }
  if (property.format === "date-time"
      && (!/^\d{4}-\d{2}-\d{2}T/.test(value) || Number.isNaN(Date.parse(value)))) {
    throw new Error(`MCP Elicitation field ${property.name} needs an ISO date-time.`);
  }
}

function nonnegativeBounds(
  value: Record<string, unknown>,
  minimumKey: string,
  maximumKey: string,
  name: string
): [number | null, number | null] {
  const minimum = optionalNonnegativeInteger(value[minimumKey], `${name} ${minimumKey}`);
  const maximum = optionalNonnegativeInteger(value[maximumKey], `${name} ${maximumKey}`);
  if (minimum !== null && maximum !== null && minimum > maximum) {
    throw new Error(`MCP Elicitation field ${name} has inconsistent size bounds.`);
  }
  return [minimum, maximum];
}

function optionalNonnegativeInteger(value: unknown, label: string): number | null {
  if (value === undefined) return null;
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw new Error(`MCP Elicitation ${label} is malformed.`);
  }
  return Number(value);
}

function optionalFiniteNumber(value: unknown, label: string): number | null {
  if (value === undefined) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`MCP Elicitation ${label} is malformed.`);
  }
  return value;
}

function optionalBoundedText(value: unknown, label: string): string | null {
  if (value === undefined) return null;
  if (typeof value !== "string" || value.length > 2_048) {
    throw new Error(`MCP Elicitation ${label} is malformed.`);
  }
  return value;
}

function stringOptions(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_OPTIONS
      || value.some((item) => typeof item !== "string" || !item || item.length > 2_048)
      || new Set(value).size !== value.length) {
    throw new Error(`MCP Elicitation field ${name} choices are malformed.`);
  }
  return [...value] as string[];
}

function titledOptions(value: unknown, name: string): ElicitationOption[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > MAX_OPTIONS) {
    throw new Error(`MCP Elicitation field ${name} choices are malformed.`);
  }
  const options = value.map((selected) => {
    if (!isRecord(selected) || !hasExactKeys(selected, ["const", "title"])
        || typeof selected.const !== "string" || !selected.const || selected.const.length > 2_048
        || typeof selected.title !== "string" || !selected.title || selected.title.length > 2_048) {
      throw new Error(`MCP Elicitation field ${name} choices are malformed.`);
    }
    return { value: selected.const, label: selected.title };
  });
  if (new Set(options.map((option) => option.value)).size !== options.length) {
    throw new Error(`MCP Elicitation field ${name} choices are duplicated.`);
  }
  return options;
}

function validInertUrl(value: string): boolean {
  if (!value || value.length > MAX_TEXT_CHARS) return false;
  try {
    const parsed = new URL(value);
    return (parsed.protocol === "http:" || parsed.protocol === "https:") && Boolean(parsed.hostname);
  } catch {
    return false;
  }
}

function primitiveDraftText(value: unknown): string {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function hasUnknownKeys(value: Record<string, unknown>, allowed: Set<string>): boolean {
  return Object.keys(value).some((key) => !allowed.has(key));
}

function hasExactKeys(value: Record<string, unknown>, expected: string[]): boolean {
  const actual = Object.keys(value).sort();
  return actual.length === expected.length
    && actual.every((key, index) => key === [...expected].sort()[index]);
}

function emptyRecord<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

function jsonByteLength(value: unknown): number {
  try {
    const encoded = JSON.stringify(value);
    if (typeof encoded !== "string") throw new Error("not JSON");
    return new TextEncoder().encode(encoded).byteLength;
  } catch {
    throw new Error("MCP Elicitation requests are not a strict JSON tree.");
  }
}
