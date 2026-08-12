export type ConfirmationRequest = {
  title: string;
  message: string;
  details: Record<string, unknown>;
  action(): Promise<void>;
  /** Erase any one-shot local material if the confirmation is discarded. */
  onCancel?(): void;
  /** Return true only when reconciliation proves the stale request was never admitted. */
  onErrorReconciled?(error: unknown): boolean;
};

export type RunGuiAction = (action: () => Promise<void>, label?: string) => Promise<boolean>;
