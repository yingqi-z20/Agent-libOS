export const runtimeDefaultDatabase = "local";

export function databaseTargetFromRenderer(value: unknown): string {
  if (value === null || value === undefined) return runtimeDefaultDatabase;
  if (typeof value !== "string") {
    throw new Error("Runtime database selection must be a string.");
  }
  const selected = value.trim();
  if (!selected || selected === runtimeDefaultDatabase) return runtimeDefaultDatabase;
  throw new Error("Runtime database paths must be selected with the Open SQLite database dialog.");
}
