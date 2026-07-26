import type { HumanRequest, ProcessMessage, ProcessOutcome, RuntimeSnapshot } from "./api/types";

export type UserConversationItem =
  | {
      id: string;
      role: "user";
      time: string;
      text: string;
      message: ProcessMessage;
    }
  | {
      id: string;
      role: "assistant";
      time: string;
      text: string;
      protected: boolean;
      request: HumanRequest;
    }
  | {
      id: string;
      role: "request";
      time: string;
      text: string;
      request: HumanRequest;
    }
  | {
      id: string;
      role: "decision";
      time: string;
      text: string;
      status: string;
      request: HumanRequest;
    }
  | {
      id: string;
      role: "terminal";
      time: string;
      text: string;
      outcome: ProcessOutcome;
    };

export function deriveUserConversation(snapshot: RuntimeSnapshot | null, pid: string | null): UserConversationItem[] {
  if (!snapshot || !pid) return [];
  const process = snapshot.processes.find((item) => item.pid === pid);
  const messages = process?.messages ?? [];
  const items: UserConversationItem[] = [];

  for (const message of messages) {
    if (!isHumanUserMessage(message)) continue;
    items.push({
      id: `message:${message.message_id}`,
      role: "user",
      time: message.created_at,
      text: message.body || message.subject || "(empty message)",
      message
    });
  }

  for (const request of snapshot.human_requests) {
    if (request.pid !== pid) continue;
    if (isHumanOutput(request)) {
      const protectedOutput = isProtectedOutput(request);
      items.push({
        id: `assistant:${request.request_id}`,
        role: "assistant",
        time: request.updated_at || request.created_at,
        text: protectedOutput ? "" : String(request.payload.message ?? ""),
        protected: protectedOutput,
        request
      });
      continue;
    }
    if (request.status === "pending") {
      items.push({
        id: `request:${request.request_id}`,
        role: "request",
        time: request.created_at,
        text: humanRequestPrompt(request),
        request
      });
      continue;
    }
    if (isHumanDecision(request)) {
      items.push({
        id: `decision:${request.request_id}`,
        role: "decision",
        time: request.updated_at || request.created_at,
        text: humanRequestDecisionText(request),
        status: request.status,
        request
      });
    }
  }

  if (process?.outcome) {
    const outcomeReference = process.outcome.kind === "killed"
      ? process.outcome.reason_oid
      : process.outcome.result_oid;
    const generatedStatus = outcomeReference
      ? `${process.outcome.kind === "killed" ? "reason_oid" : "result_oid"}:${outcomeReference}`
      : null;
    const latestConversationTime = items.reduce(
      (latest, item) => item.time.localeCompare(latest) > 0 ? item.time : latest,
      "1970-01-01T00:00:00.000Z"
    );
    items.push({
      id: `outcome:${process.pid}:${process.state_generation}:${process.outcome.kind}`,
      role: "terminal",
      time: process.updated_at || process.created_at || latestConversationTime,
      text: process.status_message === generatedStatus ? "" : process.status_message ?? "",
      outcome: process.outcome
    });
  }

  return items.sort((left, right) => left.time.localeCompare(right.time));
}

export function isHumanOutput(request: HumanRequest): boolean {
  return request.status === "delivered" && request.payload?.type === "output";
}

function isProtectedOutput(request: HumanRequest): boolean {
  const observation = request.payload.payload_observation;
  return request.payload.release_required === true || (
    typeof observation === "object"
    && observation !== null
    && "redacted" in observation
    && observation.redacted === true
  );
}

export function isHumanUserMessage(message: ProcessMessage): boolean {
  return message.sender.startsWith("human:");
}

export function isHumanDecision(request: HumanRequest): boolean {
  return request.status === "approved" || request.status === "rejected" || request.status === "edited";
}

export function humanRequestDecisionText(request: HumanRequest): string {
  const decision = request.decision ?? {};
  const answer = decision.answer;
  if (answer !== undefined && answer !== null) return String(answer);
  return "";
}

export function humanRequestPrompt(request: HumanRequest): string {
  return String(
    request.payload?.question ??
      request.payload?.reason ??
      request.payload?.type ??
      "Human input required"
  );
}
