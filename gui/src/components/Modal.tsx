import { useEffect, useId, useRef, type ReactNode } from "react";

type ModalProps = {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
  busy?: boolean;
  descriptionId?: string;
  onClose(): void;
};

const focusableSelector = [
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "a[href]",
  "[tabindex]:not([tabindex='-1'])"
].join(",");

let openModalCount = 0;
let bodyOverflowBeforeModal = "";

function lockDocumentScroll() {
  if (openModalCount === 0) {
    bodyOverflowBeforeModal = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  openModalCount += 1;

  return () => {
    openModalCount = Math.max(0, openModalCount - 1);
    if (openModalCount === 0) {
      document.body.style.overflow = bodyOverflowBeforeModal;
      bodyOverflowBeforeModal = "";
    }
  };
}

/** Accessible modal shell shared by destructive confirmations and editors. */
export function Modal({
  title,
  children,
  actions,
  className = "",
  busy = false,
  descriptionId,
  onClose
}: ModalProps) {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const unlockDocumentScroll = lockDocumentScroll();
    dialogRef.current?.focus();
    return () => {
      unlockDocumentScroll();
      previouslyFocused?.focus();
    };
  }, []);

  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape" && !busy) {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector));
    if (focusable.length === 0) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  return (
    <div
      className="modalBackdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={`modal ${className}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-busy={busy || undefined}
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
      >
        <h2 id={titleId}>{title}</h2>
        {children}
        {actions ? <div className="modalActions">{actions}</div> : null}
      </div>
    </div>
  );
}
