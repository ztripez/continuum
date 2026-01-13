import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  Executable,
} from "vscode-languageclient/node";
import * as vscode from "vscode";
import { uriExists, resolveFromWorkspace, binName } from "./helpers";

import path from "path";

export class CdslController implements vscode.Disposable {
  private client?: LanguageClient;
  private readonly disposables = new vscode.Disposable(this.stop.bind(this));
  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly log: vscode.LogOutputChannel,
  ) {}

  async start(): Promise<void> {
    // Resolve config once
    const config = vscode.workspace.getConfiguration("cdsl");
    const serverPath = config.get<string>("server.path");

    if (!serverPath) {
      this.log.error('Missing setting: "cdsl.server.path"');
      return;
    }

    const serverRoot = resolveFromWorkspace(serverPath);

    const lspFsPath = path.join(serverRoot, binName("cdsl-lsp"));
    const dapFsPath = path.join(serverRoot, binName("cdsl-dap"));

    const lspUri = vscode.Uri.file(lspFsPath);
    const dapUri = vscode.Uri.file(dapFsPath);

    this.log.info(`serverRoot=${serverRoot}`);
    this.log.info(`lsp=${lspFsPath}`);
    this.log.info(`dap=${dapFsPath}`);

    // Tear down previous run (supports reload on config change)
    await this.stop();

    // LSP
    if (await uriExists(lspUri)) {
      this.client = this.createLspClient(lspFsPath);
      this.context.subscriptions.push(this.client);
      await this.client.start();
      this.log.info("LSP started");
    } else {
      this.log.error(`Missing LSP binary: ${lspFsPath}`);
    }

    // DAP
    if (await uriExists(dapUri)) {
      const d = this.registerDap(dapFsPath);
      this.context.subscriptions.push(d);
      this.log.info("DAP registered");
    } else {
      this.log.error(`Missing DAP binary: ${dapFsPath}`);
    }
  }

  async stop(): Promise<void> {
    if (this.client) {
      await this.client.stop();
      this.client = undefined;
    }
  }

  private createLspClient(lspPath: string): LanguageClient {
    const exe: Executable = { command: lspPath, args: [] };

    const serverOptions: ServerOptions = { run: exe, debug: exe };

    const clientOptions: LanguageClientOptions = {
      documentSelector: [{ scheme: "file", language: "cdsl" }],
      outputChannel: this.log, // good
      synchronize: {
        fileEvents: vscode.workspace.createFileSystemWatcher("**/*.cdsl"),
      },
    };

    return new LanguageClient("cdsl", "CDSL Language Server", serverOptions, clientOptions);
  }

  private registerDap(dapPath: string): vscode.Disposable {
    return vscode.debug.registerDebugAdapterDescriptorFactory("cdsl", {
      createDebugAdapterDescriptor: () => {
        this.log.info(`Using Debug Adapter at ${dapPath}`);
        return new vscode.DebugAdapterExecutable(dapPath, []);
      },
    });
  }

  dispose(): void {
    void this.stop();
    this.disposables.dispose();
  }
}
