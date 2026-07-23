export type ConfirmationRequest = {
  title: string;
  message: string;
  details: Record<string, unknown>;
  action(): Promise<void>;
};

export type RunGuiAction = (action: () => Promise<void>, label?: string) => Promise<boolean>;
