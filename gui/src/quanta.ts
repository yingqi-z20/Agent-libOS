export type OptionalQuanta = number | null;
export type QuantaDraft = {
  raw: string;
  value: OptionalQuanta;
  valid: boolean;
};

export function parseQuantaDraft(value: string): QuantaDraft {
  const trimmed = value.trim();
  if (trimmed === "") return { raw: value, value: null, valid: true };
  const parsed = Number(trimmed);
  const valid = Number.isSafeInteger(parsed) && parsed > 0;
  return { raw: value, value: valid ? parsed : null, valid };
}

export function parseOptionalQuanta(value: string): OptionalQuanta {
  return parseQuantaDraft(value).value;
}
