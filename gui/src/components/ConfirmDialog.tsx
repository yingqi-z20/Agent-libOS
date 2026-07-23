import { CollapsibleJson } from "./CollapsibleJson";
import { useI18n } from "../i18n";
import { LoaderCircle } from "lucide-react";
import { useId } from "react";
import { Modal } from "./Modal";

type ConfirmDialogProps = {
  title: string;
  message: string;
  details?: Record<string, unknown>;
  confirmLabel?: string;
  busy?: boolean;
  onConfirm(): void;
  onCancel(): void;
};

export function ConfirmDialog({ title, message, details, confirmLabel, busy = false, onConfirm, onCancel }: ConfirmDialogProps) {
  const { t } = useI18n();
  const descriptionId = useId();
  return (
    <Modal
      title={title}
      busy={busy}
      descriptionId={descriptionId}
      onClose={onCancel}
      actions={
        <>
          <button className="secondary" disabled={busy} onClick={onCancel}>{t("confirm.cancel")}</button>
          <button className="danger" disabled={busy} onClick={onConfirm}>
            {busy ? <LoaderCircle className="spin" size={15} aria-hidden="true" /> : null}
            {busy ? t("confirm.working") : confirmLabel ?? t("confirm.confirm")}
          </button>
        </>
      }
    >
      <p id={descriptionId}>{message}</p>
      {details ? <CollapsibleJson value={details} label={t("confirm.preview")} /> : null}
    </Modal>
  );
}
