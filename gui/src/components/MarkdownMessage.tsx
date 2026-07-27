import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { MouseEvent, ReactNode } from "react";

const SAFE_MARKDOWN_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

const markdownComponents: Components = {
  img({ alt }) {
    const accessibleLabel = alt || "Image omitted";
    return (
      <span className="markdownImagePlaceholder" role="img" aria-label={accessibleLabel}>
        {alt ? `[image: ${alt}]` : "[image omitted]"}
      </span>
    );
  },
  a({ href, children, node: _node, ...props }) {
    if (!isSafeMarkdownHref(href)) {
      return <span>{children}</span>;
    }
    return (
      <a
        {...props}
        href={href}
        rel="noreferrer noopener"
        onClick={(event) => openMarkdownHref(href, event)}
      >
        {children}
      </a>
    );
  },
  table({ children, node: _node, ...props }) {
    return (
      <div className="markdownTableWrap">
        <table {...props}>{children}</table>
      </div>
    );
  }
};

export function MarkdownMessage({ text, fallback }: { text: string; fallback: ReactNode }) {
  const value = text || String(fallback ?? "");
  return (
    <div className="markdownMessage">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {value}
      </ReactMarkdown>
    </div>
  );
}

export function isSafeMarkdownHref(href: string | undefined): href is string {
  if (!href) return false;
  try {
    const parsed = new URL(href);
    return SAFE_MARKDOWN_PROTOCOLS.has(parsed.protocol.toLowerCase());
  } catch {
    return false;
  }
}

export function openMarkdownHref(
  href: string,
  event?: Pick<MouseEvent<HTMLAnchorElement>, "preventDefault">
): boolean {
  if (!isSafeMarkdownHref(href)) return false;
  event?.preventDefault();
  void window.libosApi?.openExternal(href);
  return true;
}
