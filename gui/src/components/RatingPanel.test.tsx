import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RuntimeProcess } from "../api/types";
import { I18nProvider } from "../i18n";
import { RatingPanel, ratingValueForKey } from "./RatingPanel";

describe("RatingPanel", () => {
  it("renders an existing process rating", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <RatingPanel process={processWithRating()} onSave={() => true} />
      </I18nProvider>
    );

    expect(html).toMatch(/Rating|评分/);
    expect(html).toMatch(/Saved by owner|owner 已保存/);
    expect(html).toContain("well handled");
    expect(html).toContain("aria-checked=\"true\"");
    expect(html).toMatch(/4\/5/);
  });

  it("disables controls when no process is selected", () => {
    const html = renderToStaticMarkup(
      <I18nProvider>
        <RatingPanel process={null} onSave={() => true} />
      </I18nProvider>
    );

    expect(html).toMatch(/Not rated yet|尚未评分/);
    expect(html).toContain("disabled=\"\"");
    expect(html).toMatch(/Choose a score|选择分数/);
  });

  it("supports wrapped radio-group keyboard navigation", () => {
    expect(ratingValueForKey(5, "ArrowRight")).toBe(1);
    expect(ratingValueForKey(1, "ArrowLeft")).toBe(5);
    expect(ratingValueForKey(3, "Home")).toBe(1);
    expect(ratingValueForKey(3, "End")).toBe(5);
    expect(ratingValueForKey(3, "Enter")).toBeNull();
  });
});

function processWithRating(): RuntimeProcess {
  return {
    pid: "pid_1",
    parent_pid: null,
    image_id: "coding-agent:v0",
    llm_profile_id: "default",
    status: "exited",
    goal_oid: null,
    checkpoint_head: null,
    working_directory: ".",
    status_message: null,
    wait_state: null,
    outcome: null,
    state_generation: 0,
    loaded_skills: {},
    tool_table: {},
    capabilities: [],
    terminal: true,
    unread_message_count: 0,
    interrupt_count: 0,
    messages: [],
    llm_call_count: 0,
    token_total: 0,
    resource_budget: {},
    resource_usage: {},
    resource_remaining: {},
    rating: {
      rating_id: "rating_1",
      pid: "pid_1",
      score: 4,
      comment: "well handled",
      rater: "owner",
      source: "gui",
      metadata: {},
      created_at: "2026-06-29T01:00:00Z",
      updated_at: "2026-06-29T01:00:00Z"
    }
  };
}
