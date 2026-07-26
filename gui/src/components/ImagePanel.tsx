import { Download, Eye, LoaderCircle, Save } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ImageInspectResult, ImageSummary, RuntimeProcess } from "../api/types";
import { useI18n } from "../i18n";
import { RequestEpoch } from "../requestEpoch";
import { CollapsibleJson } from "./CollapsibleJson";

type ImageCommitRequest = {
  imageId: string;
  name: string;
  version: string;
  replace: boolean;
  checkpointId?: string;
};

type ImagePanelProps = {
  images: ImageSummary[];
  selectedProcess: RuntimeProcess | null;
  compact?: boolean;
  allowReplace?: boolean;
  onImportImage(replace: boolean): void;
  onCommitImage(request: ImageCommitRequest): void;
  onUseForSpawn?(imageId: string): void;
  onUseForExec?(imageId: string): void;
  onInspectImage?(imageId: string): Promise<ImageInspectResult>;
};

export function ImagePanel({
  images,
  selectedProcess,
  compact = false,
  allowReplace = false,
  onImportImage,
  onCommitImage,
  onUseForSpawn,
  onUseForExec,
  onInspectImage
}: ImagePanelProps) {
  const { t } = useI18n();
  const [replace, setReplace] = useState(false);
  const [checkpointId, setCheckpointId] = useState("");
  const [imageId, setImageId] = useState("");
  const [name, setName] = useState("");
  const [version, setVersion] = useState("v0");
  const [inspected, setInspected] = useState<ImageInspectResult | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [inspectingId, setInspectingId] = useState<string | null>(null);
  const inspectRequests = useRef(new RequestEpoch());

  useEffect(() => () => inspectRequests.current.invalidate(), []);

  const commitDisabled = !selectedProcess || !imageId.trim() || !name.trim() || !version.trim();

  async function inspect(image: ImageSummary) {
    if (!onInspectImage) return;
    const request = inspectRequests.current.begin();
    setInspectError(null);
    setInspectingId(image.image_id);
    try {
      const result = await onInspectImage(image.image_id);
      if (inspectRequests.current.isCurrent(request)) setInspected(result);
    } catch (error) {
      if (inspectRequests.current.isCurrent(request)) {
        setInspectError(error instanceof Error ? error.message : String(error));
      }
    } finally {
      if (inspectRequests.current.isCurrent(request)) setInspectingId(null);
    }
  }

  return (
    <section className={compact ? "imagePanel compact" : "imagePanel"}>
      <div className="imagePanelActions">
        <button onClick={() => onImportImage(allowReplace && replace)}><Download size={14} />{t("image.import")}</button>
        {allowReplace ? (
          <label className="toggle inlineToggle">
            <input type="checkbox" checked={replace} onChange={(event) => setReplace(event.currentTarget.checked)} />
            {t("image.replace")}
          </label>
        ) : null}
      </div>

      <div className="imageCommitBox">
        <input aria-label={t("image.commitIdPlaceholder")} value={imageId} onChange={(event) => setImageId(event.currentTarget.value)} placeholder={t("image.commitIdPlaceholder")} />
        <input aria-label={t("image.commitNamePlaceholder")} value={name} onChange={(event) => setName(event.currentTarget.value)} placeholder={t("image.commitNamePlaceholder")} />
        <input aria-label={t("image.version")} value={version} onChange={(event) => setVersion(event.currentTarget.value)} placeholder={t("image.version")} />
        {allowReplace ? (
          <input aria-label={t("image.checkpointPlaceholder")} value={checkpointId} onChange={(event) => setCheckpointId(event.currentTarget.value)} placeholder={t("image.checkpointPlaceholder")} />
        ) : null}
        <button
          className="warning"
          disabled={commitDisabled}
          onClick={() => onCommitImage({
            imageId: imageId.trim(),
            name: name.trim(),
            version: version.trim(),
            replace: allowReplace && replace,
            checkpointId: checkpointId.trim() || undefined
          })}
        >
          <Save size={14} />{t("image.save")}
        </button>
      </div>

      <div className="imageList">
        {images.length === 0 ? <div className="empty">{t("image.empty")}</div> : null}
        {images.map((image) => (
          <article className="imageRow" key={image.image_id}>
            <div>
              <strong>{image.image_id}</strong>
              <span>{image.name} · {image.version} · {image.boot_kind}</span>
            </div>
            <span>
              {t("image.requiredCaps", { count: image.required_capabilities_count })}
              {" · "}
              {t("image.requiredModules", { count: image.required_modules_count })}
            </span>
            <div className="imageRowActions">
              {onUseForSpawn ? <button aria-label={`${t("image.useForSpawn")}: ${image.image_id}`} onClick={() => onUseForSpawn(image.image_id)}>{t("image.useForSpawn")}</button> : null}
              {onUseForExec ? <button aria-label={`${t("image.useForExec")}: ${image.image_id}`} onClick={() => onUseForExec(image.image_id)}>{t("image.useForExec")}</button> : null}
              {onInspectImage ? (
                <button
                  aria-label={`${t("image.inspect")}: ${image.image_id}`}
                  aria-busy={inspectingId === image.image_id || undefined}
                  disabled={inspectingId !== null}
                  onClick={() => void inspect(image)}
                >
                  {inspectingId === image.image_id
                    ? <LoaderCircle className="spin" size={14} aria-hidden="true" />
                    : <Eye size={14} aria-hidden="true" />}
                  {inspectingId === image.image_id ? t("user.working") : t("image.inspect")}
                </button>
              ) : null}
            </div>
          </article>
        ))}
      </div>

      {inspectError ? <div className="toast inlineToast" role="alert">{inspectError}</div> : null}
      {inspected ? <CollapsibleJson value={inspected} label={t("image.inspectResult")} /> : null}
    </section>
  );
}
