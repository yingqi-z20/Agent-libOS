import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("libosApi", {
  getConnection: () => ipcRenderer.invoke("libos:getConnection"),
  chooseDatabase: () => ipcRenderer.invoke("libos:chooseDatabase"),
  chooseImagePackage: () => ipcRenderer.invoke("libos:chooseImagePackage"),
  openExternal: (url: string) => ipcRenderer.invoke("libos:openExternal", url)
});
