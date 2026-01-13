import * as vscode from "vscode";
import { CdslController } from "./controller";
let controller: CdslController | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const log = vscode.window.createOutputChannel("Continuum", { log: true });
  context.subscriptions.push(log);

  controller = new CdslController(context, log);
  context.subscriptions.push(controller);

  // initial start
  await controller.start();

  // reload on config changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration(async (e) => {
      if (e.affectsConfiguration("cdsl.server.path")) {
        log.info("cdsl.server.path changed, reloading...");
        if (controller) {
          await controller.start();
        }
      }
    })
  );
}

export function deactivate(): Thenable<void> | undefined {
  return controller?.stop();
}
