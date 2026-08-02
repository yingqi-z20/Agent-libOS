/// <reference types="vite/client" />

export type GuiConnection = {
  url: string;
  token: string;
  db: string;
};

export type ImagePackageFile = {
  name: string;
  manifest: string;
  manifest_sha256: string;
  files: Record<string, string | { base64: string }>;
};

declare global {
  interface Window {
    libosApi?: {
      getConnection(): Promise<GuiConnection | null>;
      chooseDatabase(): Promise<GuiConnection | null>;
      chooseImagePackage(): Promise<ImagePackageFile | null>;
      openExternal(url: string): Promise<boolean>;
    };
  }
}
