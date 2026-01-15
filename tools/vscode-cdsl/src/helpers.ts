import * as path from "path";
import * as vscode from "vscode";

export function workspaceRoot(): string | undefined {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
}

export function resolveFromWorkspace(p: string): string {
  if (path.isAbsolute(p)) return p;
  const ws = workspaceRoot();
  return ws ? path.resolve(ws, p) : path.resolve(p);
}

export async function uriExists(uri: vscode.Uri): Promise<boolean> {
  try {
    await vscode.workspace.fs.stat(uri);
    return true;
  } catch {
    return false;
  }
}

export function binName(name: string): string {
  return process.platform === "win32" ? `${name}.exe` : name;
}
