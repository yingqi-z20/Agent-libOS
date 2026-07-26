import { useEffect, useState } from "react";
import { Star } from "lucide-react";
import type { RuntimeProcess } from "../api/types";
import { useI18n } from "../i18n";

type RatingPanelProps = {
  process: RuntimeProcess | null;
  onSave(pid: string, score: number, comment: string): Promise<boolean> | boolean;
};

export function RatingPanel({ process, onSave }: RatingPanelProps) {
  const { t } = useI18n();
  const [score, setScore] = useState(process?.rating?.score ?? 0);
  const [comment, setComment] = useState(process?.rating?.comment ?? "");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setScore(process?.rating?.score ?? 0);
    setComment(process?.rating?.comment ?? "");
  }, [process?.pid, process?.rating?.score, process?.rating?.comment]);

  async function save() {
    if (!process || score < 1 || saving) return;
    setSaving(true);
    try {
      await onSave(process.pid, score, comment);
    } finally {
      setSaving(false);
    }
  }

  const disabled = !process || saving;
  return (
    <section className="agentRatingPanel" aria-label={t("rating.label")}>
      <div className="ratingHeader">
        <strong>{t("rating.title")}</strong>
        {process?.rating ? (
          <span>{t("rating.savedBy", { rater: process.rating.rater })}</span>
        ) : (
          <span>{t("rating.notRated")}</span>
        )}
      </div>
      <div className="ratingStars" role="radiogroup" aria-label={t("rating.scoreLabel")}>
        {[1, 2, 3, 4, 5].map((value) => (
          <button
            type="button"
            key={value}
            className={value <= score ? "active" : ""}
            disabled={disabled}
            role="radio"
            aria-checked={score === value}
            aria-label={t("rating.starLabel", { score: value })}
            tabIndex={(score || 1) === value ? 0 : -1}
            onClick={() => setScore(value)}
            onKeyDown={(event) => {
              const next = ratingValueForKey(value, event.key);
              if (next === null) return;
              event.preventDefault();
              setScore(next);
              const radios = event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>("[role='radio']");
              radios?.[next - 1]?.focus();
            }}
          >
            <Star size={17} fill={value <= score ? "currentColor" : "none"} />
          </button>
        ))}
      </div>
      <textarea
        value={comment}
        disabled={disabled}
        aria-label={t("rating.commentPlaceholder")}
        placeholder={t("rating.commentPlaceholder")}
        onChange={(event) => setComment(event.currentTarget.value)}
      />
      <div className="ratingActions">
        <span>{score > 0 ? t("rating.currentScore", { score }) : t("rating.chooseScore")}</span>
        <button type="button" className="primary" disabled={!process || score < 1 || saving} onClick={() => void save()}>
          {saving ? t("rating.saving") : t("rating.save")}
        </button>
      </div>
    </section>
  );
}

export function ratingValueForKey(current: number, key: string): number | null {
  if (key === "Home") return 1;
  if (key === "End") return 5;
  if (key === "ArrowRight" || key === "ArrowUp") return current === 5 ? 1 : current + 1;
  if (key === "ArrowLeft" || key === "ArrowDown") return current === 1 ? 5 : current - 1;
  return null;
}
