import type { ImageSummary } from "../api/types";
import { useId } from "react";
import { useI18n } from "../i18n";

type ImageSelectProps = {
  images: ImageSummary[];
  value: string;
  label?: string;
  disabled?: boolean;
  onChange(value: string): void;
};

export function ImageSelect({ images, value, label, disabled = false, onChange }: ImageSelectProps) {
  const { t } = useI18n();
  const labelId = useId();
  const known = images.some((image) => image.image_id === value);
  const controlLabel = label ?? t("image.selectLabel");
  return (
    <div className="imageSelect" role="group" aria-labelledby={labelId}>
      <span id={labelId}>{controlLabel}</span>
      <select aria-labelledby={labelId} value={known ? value : ""} disabled={disabled} onChange={(event) => onChange(event.currentTarget.value)}>
        {!known ? <option value="">{t("image.customOption")}</option> : null}
        {images.map((image) => (
          <option key={image.image_id} value={image.image_id}>
            {image.image_id} · {image.boot_kind}
          </option>
        ))}
      </select>
      <input
        aria-label={`${controlLabel}: ${t("image.manualPlaceholder")}`}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.currentTarget.value)}
        placeholder={t("image.manualPlaceholder")}
      />
    </div>
  );
}
