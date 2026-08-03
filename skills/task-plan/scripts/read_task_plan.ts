type JsonRecord = { [key: string]: unknown };
type PlanStatus =
  | "pending"
  | "in_progress"
  | "blocked"
  | "completed"
  | "cancelled";
type PlanStep = { step: string; status: PlanStatus };
type Snapshot = {
  revision: number;
  explanation: string | null;
  plan: PlanStep[];
};
type StatusCounts = Record<PlanStatus, number>;
type ParsedLedger = { entries: Snapshot[]; latest: Snapshot };
type LibOS = {
  syscall(name: string, args: JsonRecord): Promise<unknown>;
};

const SCHEMA_VERSION = "task-plan/v1";
const STATUS_VALUES: string[] = [
  "pending",
  "in_progress",
  "blocked",
  "completed",
  "cancelled",
];

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, label: string): JsonRecord {
  if (!isRecord(value)) {
    throw new Error(label + " must be an object");
  }
  return value;
}

function requireExactKeys(
  value: JsonRecord,
  expected: string[],
  label: string,
): void {
  const actual = Object.keys(value).sort();
  const wanted = expected.slice().sort();
  if (
    actual.length !== wanted.length ||
    actual.some((item, index) => item !== wanted[index])
  ) {
    throw new Error(label + " has an invalid shape");
  }
}

function normalizeName(value: unknown): string {
  if (typeof value !== "string") {
    throw new Error("name must be a string");
  }
  const normalized = value.trim();
  if (normalized.length < 1 || normalized.length > 128) {
    throw new Error("name must contain 1 to 128 characters");
  }
  return normalized;
}

function normalizeNamespace(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error("namespace must be a string or null");
  }
  const normalized = value.trim();
  if (normalized.length < 1 || normalized.length > 512) {
    throw new Error("namespace must contain 1 to 512 characters");
  }
  return normalized;
}

function normalizeStoredExplanation(value: unknown, label: string): string | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error(label + " must be a string or null");
  }
  const normalized = value.trim();
  if (
    normalized.length < 1 ||
    normalized.length > 2048 ||
    normalized !== value
  ) {
    throw new Error(label + " is not normalized");
  }
  return normalized;
}

function normalizeStoredPlan(value: unknown, label: string): PlanStep[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 32) {
    throw new Error(label + " must contain 1 to 32 steps");
  }
  let active = 0;
  const plan = value.map((item, index): PlanStep => {
    const itemLabel = label + "[" + index + "]";
    const stepRecord = requireRecord(item, itemLabel);
    requireExactKeys(stepRecord, ["step", "status"], itemLabel);
    if (typeof stepRecord.step !== "string") {
      throw new Error(itemLabel + ".step must be a string");
    }
    const step = stepRecord.step.trim();
    if (
      step.length < 1 ||
      step.length > 512 ||
      step !== stepRecord.step
    ) {
      throw new Error(itemLabel + ".step is not normalized");
    }
    if (
      typeof stepRecord.status !== "string" ||
      !STATUS_VALUES.includes(stepRecord.status)
    ) {
      throw new Error(itemLabel + ".status is invalid");
    }
    const status = stepRecord.status as PlanStatus;
    if (status === "in_progress") {
      active += 1;
    }
    return { step, status };
  });
  if (active > 1) {
    throw new Error(label + " contains more than one in_progress step");
  }
  return plan;
}

function requirePositiveInteger(value: unknown, label: string): number {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < 1
  ) {
    throw new Error(label + " must be a positive integer");
  }
  return value;
}

function parseLedger(observed: JsonRecord): ParsedLedger {
  if (observed.type !== "plan") {
    throw new Error("Object Memory object is not type plan");
  }
  const payload = requireRecord(observed.payload, "plan payload");
  requireExactKeys(payload, ["schema_version", "entries"], "plan payload");
  if (payload.schema_version !== SCHEMA_VERSION) {
    throw new Error("unsupported task plan schema_version");
  }
  if (!Array.isArray(payload.entries) || payload.entries.length < 1) {
    throw new Error("plan payload entries must be a non-empty array");
  }
  const entries = payload.entries.map((item, index): Snapshot => {
    const label = "plan payload entries[" + index + "]";
    const entry = requireRecord(item, label);
    requireExactKeys(entry, ["revision", "explanation", "plan"], label);
    const revision = requirePositiveInteger(entry.revision, label + ".revision");
    if (revision !== index + 1) {
      throw new Error("plan revisions must be contiguous starting at 1");
    }
    return {
      revision,
      explanation: normalizeStoredExplanation(
        entry.explanation,
        label + ".explanation",
      ),
      plan: normalizeStoredPlan(entry.plan, label + ".plan"),
    };
  });
  return { entries, latest: entries[entries.length - 1] };
}

function statusCounts(plan: PlanStep[]): StatusCounts {
  const counts: StatusCounts = {
    pending: 0,
    in_progress: 0,
    blocked: 0,
    completed: 0,
    cancelled: 0,
  };
  for (const item of plan) {
    counts[item.status] += 1;
  }
  return counts;
}

function requireResultText(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error("memory.read_object returned invalid " + label);
  }
  return value;
}

export async function run(
  args: JsonRecord,
  libos: LibOS,
): Promise<JsonRecord> {
  const name = normalizeName(args.name);
  const namespace = normalizeNamespace(args.namespace);
  const observed = requireRecord(
    await libos.syscall("memory.read_object", { name, namespace }),
    "memory.read_object result",
  );
  const ledger = parseLedger(observed);
  return {
    oid: requireResultText(observed.oid, "oid"),
    namespace: requireResultText(observed.namespace, "namespace"),
    name: requireResultText(observed.name, "name"),
    memory_version: requirePositiveInteger(
      observed.version,
      "memory.read_object version",
    ),
    revision: ledger.latest.revision,
    explanation: ledger.latest.explanation,
    plan: ledger.latest.plan,
    status_counts: statusCounts(ledger.latest.plan),
  };
}
