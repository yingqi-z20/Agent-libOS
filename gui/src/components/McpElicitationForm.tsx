import type { McpInputRequest } from "../api/types";
import {
  inspectElicitationRequests,
  type ElicitationDrafts,
  type ElicitationFieldDraft,
  type ElicitationPlan,
  type ElicitationProperty,
  type ElicitationRequestDraft
} from "./mcpElicitation";

export function McpElicitationForm({
  requests,
  drafts,
  onChange,
  disabled = false
}: {
  requests: McpInputRequest[];
  drafts: ElicitationDrafts;
  onChange(value: ElicitationDrafts): void;
  disabled?: boolean;
}) {
  const inspection = inspectElicitationRequests(requests);
  if (inspection.error) {
    return (
      <div className="inlineError" role="alert">
        Unsupported MCP Elicitation schema. The answer is fail-closed: {inspection.error}
      </div>
    );
  }
  return (
    <div className="mcpElicitationForms" aria-label="MCP Elicitation response form">
      {inspection.plans.length === 0 ? (
        <p>Provider supplied request state without an input schema. The explicit response is the empty object <code>{"{}"}</code>.</p>
      ) : null}
      {inspection.plans.map((plan) => (
        <RequestForm
          key={plan.requestId}
          plan={plan}
          draft={drafts[plan.requestId]}
          disabled={disabled}
          onChange={(next) => onChange(replaceRequestDraft(drafts, plan.requestId, next))}
        />
      ))}
    </div>
  );
}

function RequestForm({
  plan,
  draft,
  disabled,
  onChange
}: {
  plan: ElicitationPlan;
  draft: ElicitationRequestDraft | undefined;
  disabled: boolean;
  onChange(value: ElicitationRequestDraft): void;
}) {
  if (!draft) {
    return <div className="inlineError" role="alert">MCP Elicitation draft is unavailable.</div>;
  }
  return (
    <fieldset className="mcpSchemaArguments">
      <legend><code>{plan.requestId}</code> — {plan.prompt}</legend>
      <label className="fieldStack spanAll">
        <span>{plan.requestId} response action</span>
        <select
          value={draft.action}
          disabled={disabled}
          onChange={(event) => onChange({
            ...draft,
            action: responseAction(event.currentTarget.value)
          })}
        >
          <option value="">Choose an explicit action…</option>
          <option value="accept">accept</option>
          <option value="decline">decline</option>
          <option value="cancel">cancel</option>
        </select>
      </label>
      {plan.mode === "url" ? (
        <UrlReview plan={plan} draft={draft} disabled={disabled} onChange={onChange} />
      ) : draft.action === "accept" ? (
        plan.properties.length ? plan.properties.map((property) => (
          <PropertyField
            key={property.name}
            requestId={plan.requestId}
            property={property}
            draft={draft.fields[property.name]}
            disabled={disabled}
            onChange={(next) => onChange({
              ...draft,
              fields: replaceFieldDraft(draft.fields, property.name, next)
            })}
          />
        )) : <p>This form requests an explicit empty object.</p>
      ) : null}
    </fieldset>
  );
}

function UrlReview({
  plan,
  draft,
  disabled,
  onChange
}: {
  plan: ElicitationPlan;
  draft: ElicitationRequestDraft;
  disabled: boolean;
  onChange(value: ElicitationRequestDraft): void;
}) {
  return (
    <div className="fieldStack spanAll">
      <span>Untrusted inert URL (never opened automatically)</span>
      <code className="codeInput">{plan.inertUrl}</code>
      <label className="toggle">
        <input
          type="checkbox"
          checked={draft.urlReviewed}
          disabled={disabled}
          onChange={(event) => onChange({ ...draft, urlReviewed: event.currentTarget.checked })}
        />
        I explicitly reviewed the inert URL for {plan.requestId}
      </label>
      {draft.action === "accept" && !draft.urlReviewed
        ? <p>Accept remains unavailable until this explicit review is recorded.</p>
        : null}
    </div>
  );
}

function PropertyField({
  requestId,
  property,
  draft,
  disabled,
  onChange
}: {
  requestId: string;
  property: ElicitationProperty;
  draft: ElicitationFieldDraft | undefined;
  disabled: boolean;
  onChange(value: ElicitationFieldDraft): void;
}) {
  if (!draft) {
    return <div className="inlineError" role="alert">MCP Elicitation field {property.name} is unavailable.</div>;
  }
  const fieldDisabled = disabled || (!property.required && !draft.included);
  const label = `${requestId} ${property.label}${property.required ? " *" : ""}`;
  return (
    <div className="fieldStack">
      {!property.required ? (
        <label className="toggle">
          <input
            type="checkbox"
            checked={draft.included}
            disabled={disabled}
            onChange={(event) => onChange({ ...draft, included: event.currentTarget.checked })}
          />
          Include optional field {property.label}
        </label>
      ) : null}
      <SchemaValueField
        label={label}
        property={property}
        draft={draft}
        disabled={fieldDisabled}
        onChange={onChange}
      />
      {property.description ? <small>{property.description}</small> : null}
    </div>
  );
}

function SchemaValueField({
  label,
  property,
  draft,
  disabled,
  onChange
}: {
  label: string;
  property: ElicitationProperty;
  draft: ElicitationFieldDraft;
  disabled: boolean;
  onChange(value: ElicitationFieldDraft): void;
}) {
  if (property.kind === "array") {
    return (
      <fieldset disabled={disabled}>
        <legend>{label}</legend>
        {property.options.map((option) => (
          <label className="toggle" key={option.value}>
            <input
              type="checkbox"
              checked={draft.selected.includes(option.value)}
              onChange={(event) => onChange({
                ...draft,
                selected: event.currentTarget.checked
                  ? [...draft.selected, option.value]
                  : draft.selected.filter((value) => value !== option.value)
              })}
            />
            {option.label}
          </label>
        ))}
      </fieldset>
    );
  }
  if (property.kind === "boolean" || property.options.length) {
    return (
      <label className="fieldStack">
        <span>{label}</span>
        <select
          value={draft.raw}
          disabled={disabled}
          onChange={(event) => onChange({ ...draft, raw: event.currentTarget.value })}
        >
          <option value="">Select…</option>
          {property.kind === "boolean" ? (
            <>
              <option value="true">true</option>
              <option value="false">false</option>
            </>
          ) : property.options.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </label>
    );
  }
  const inputType = property.kind === "number" || property.kind === "integer"
    ? "number"
    : property.format === "email"
      ? "email"
      : property.format === "date"
        ? "date"
        : property.format === "date-time"
          ? "datetime-local"
          : "text";
  return (
    <label className="fieldStack">
      <span>{label}</span>
      <input
        type={inputType}
        value={draft.raw}
        disabled={disabled}
        min={property.minimum ?? property.minimumSize ?? undefined}
        max={property.maximum ?? property.maximumSize ?? undefined}
        minLength={property.kind === "string" ? property.minimumSize ?? undefined : undefined}
        maxLength={property.kind === "string" ? property.maximumSize ?? 8_192 : undefined}
        step={property.kind === "integer" ? 1 : property.kind === "number" ? "any" : undefined}
        onChange={(event) => onChange({ ...draft, raw: event.currentTarget.value })}
      />
    </label>
  );
}

function responseAction(value: string): ElicitationRequestDraft["action"] {
  if (value === "" || value === "accept" || value === "decline" || value === "cancel") return value;
  throw new Error("MCP Elicitation response action is unsupported.");
}

function replaceRequestDraft(
  current: ElicitationDrafts,
  requestId: string,
  value: ElicitationRequestDraft
): ElicitationDrafts {
  return { ...current, [requestId]: value };
}

function replaceFieldDraft(
  current: Record<string, ElicitationFieldDraft>,
  name: string,
  value: ElicitationFieldDraft
): Record<string, ElicitationFieldDraft> {
  return { ...current, [name]: value };
}
