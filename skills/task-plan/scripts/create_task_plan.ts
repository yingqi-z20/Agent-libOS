type JsonRecord = { [key: string]: unknown };
type PlanStatus =
  | "pending"
  | "in_progress"
  | "blocked"
  | "completed"
  | "cancelled";
type PlanStep = { step: string; status: PlanStatus };
type StatusCounts = Record<PlanStatus, number>;
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

function normalizeExplanation(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error("explanation must be a string or null");
  }
  const normalized = value.trim();
  if (normalized.length > 2048) {
    throw new Error("explanation must not exceed 2048 characters");
  }
  return normalized.length === 0 ? null : normalized;
}

function normalizePlan(value: unknown): PlanStep[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 32) {
    throw new Error("plan must contain 1 to 32 steps");
  }
  let active = 0;
  const plan = value.map((item, index): PlanStep => {
    const stepRecord = requireRecord(item, "plan[" + index + "]");
    requireExactKeys(stepRecord, ["step", "status"], "plan[" + index + "]");
    if (typeof stepRecord.step !== "string") {
      throw new Error("plan[" + index + "].step must be a string");
    }
    const step = stepRecord.step.trim();
    if (step.length < 1 || step.length > 512) {
      throw new Error(
        "plan[" + index + "].step must contain 1 to 512 characters",
      );
    }
    if (
      typeof stepRecord.status !== "string" ||
      !STATUS_VALUES.includes(stepRecord.status)
    ) {
      throw new Error("plan[" + index + "].status is invalid");
    }
    const status = stepRecord.status as PlanStatus;
    if (status === "in_progress") {
      active += 1;
    }
    return { step, status };
  });
  if (active > 1) {
    throw new Error("plan may contain at most one in_progress step");
  }
  return plan;
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
    throw new Error("memory.create_object returned invalid " + label);
  }
  return value;
}

export async function run(
  args: JsonRecord,
  libos: LibOS,
): Promise<JsonRecord> {
  const name = normalizeName(args.name);
  const namespace = normalizeNamespace(args.namespace);
  const explanation = normalizeExplanation(args.explanation);
  const plan = normalizePlan(args.plan);
  const snapshot = { revision: 1, explanation, plan };
  const created = requireRecord(
    await libos.syscall("memory.create_object", {
      type: "plan",
      payload: {
        schema_version: SCHEMA_VERSION,
        entries: [snapshot],
      },
      metadata: {
        title: name,
        summary: "Revisioned task plan managed by the task-plan Skill.",
        tags: ["task-plan"],
      },
      immutable: false,
      name,
      namespace,
    }),
    "memory.create_object result",
  );
  if (created.type !== "plan") {
    throw new Error("memory.create_object returned an unexpected Object type");
  }
  return {
    created: true,
    oid: requireResultText(created.oid, "oid"),
    namespace: requireResultText(created.namespace, "namespace"),
    name: requireResultText(created.name, "name"),
    revision: 1,
    explanation,
    plan,
    status_counts: statusCounts(plan),
  };
}
