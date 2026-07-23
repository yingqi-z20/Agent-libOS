import type { ImageSummary } from "../api/types";
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
  const known = images.some((image) => image.image_id === value);
  return (
    <label className="imageSelect">
      <span>{label ?? t("image.selectLabel")}</span>
      <select value={known ? value : ""} disabled={disabled} onChange={(event) => onChange(event.currentTarget.value)}>
        {!known ? <option value="">{t("image.customOption")}</option> : null}
        {images.map((image) => (
          <option key={image.image_id} value={image.image_id}>
            {image.image_id} · {image.boot_kind}
          </option>
        ))}
      </select>
      <input value={value} disabled={disabled} onChange={(event) => onChange(event.currentTarget.value)} placeholder={t("image.manualPlaceholder")} />
    </label>
  );
}
