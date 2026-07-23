import type { RuntimeProcess } from "../api/types";
import { useI18n } from "../i18n";
import { CollapsibleJson } from "./CollapsibleJson";

export function ProcessOverview({ process }: { process: RuntimeProcess }) {
  const { t } = useI18n();
  const facts = [
    [t("overview.status"), process.status],
    [t("overview.image"), process.image_id],
    [t("overview.model"), process.llm_profile_id],
    [t("overview.cwd"), process.working_directory],
    [t("overview.parent"), process.parent_pid ?? t("overview.none")],
    [t("overview.checkpoint"), process.checkpoint_head ?? t("overview.none")],
    [t("overview.stateGeneration"), String(process.state_generation)],
    [t("overview.llmCalls"), String(process.llm_call_count)],
    [t("overview.tokens"), String(process.token_total)]
  ];
  return (
    <div className="processOverview">
      <dl className="factGrid">
        {facts.map(([label, value]) => (
          <div className="fact" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {process.status_message ? <p className="statusMessage">{process.status_message}</p> : null}
      <section className="resourceGrid" aria-label={t("overview.resources")}>
        <ResourceCard title={t("overview.budget")} value={process.resource_budget ?? {}} />
        <ResourceCard title={t("overview.usage")} value={process.resource_usage ?? {}} />
        <ResourceCard title={t("overview.remaining")} value={process.resource_remaining ?? {}} />
      </section>
      <CollapsibleJson value={process} label={t("details.rawData")} />
    </div>
  );
}

function ResourceCard({ title, value }: { title: string; value: Record<string, unknown> }) {
  return (
    <article className="resourceCard">
      <h3>{title}</h3>
      {Object.keys(value).length ? <CollapsibleJson value={value} /> : <span>—</span>}
    </article>
  );
}
