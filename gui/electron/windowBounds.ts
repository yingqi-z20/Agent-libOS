export const mainWindowBounds = Object.freeze({
  width: 1440,
  height: 920,
  minWidth: 360,
  minHeight: 520
});

export function shouldCreateBrowserWindow(smokeMode: boolean, smokeWindowMode: boolean): boolean {
  return !smokeMode || smokeWindowMode;
}
