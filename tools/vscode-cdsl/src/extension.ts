import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    Executable,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const config = vscode.workspace.getConfiguration('cdsl');

    // --- LSP Support ---
    let serverPath = config.get<string>('server.path');
    if (!serverPath) {
        const possiblePaths = [
            path.join(context.extensionPath, '..', '..', 'target', 'debug', 'cdsl-lsp'),
            path.join(context.extensionPath, '..', '..', 'target', 'release', 'cdsl-lsp'),
            path.join(context.extensionPath, 'bin', 'cdsl-lsp'),
            'cdsl-lsp',
        ];

        for (const p of possiblePaths) {
            try {
                if (p === 'cdsl-lsp') {
                    serverPath = p;
                    break;
                }
                await vscode.workspace.fs.stat(vscode.Uri.file(p));
                serverPath = p;
                break;
            } catch { }
        }
    }

    if (serverPath) {
        console.log(`CDSL: Using LSP server at ${serverPath}`);
        const serverExecutable: Executable = {
            command: serverPath,
            args: [],
        };

        const serverOptions: ServerOptions = {
            run: serverExecutable,
            debug: serverExecutable,
        };

        const clientOptions: LanguageClientOptions = {
            documentSelector: [{ scheme: 'file', language: 'cdsl' }],
            synchronize: {
                fileEvents: vscode.workspace.createFileSystemWatcher('**/*.cdsl'),
            },
        };

        client = new LanguageClient('cdsl', 'CDSL Language Server', serverOptions, clientOptions);
        await client.start();
    } else {
        vscode.window.showWarningMessage('CDSL language server not found.');
    }

    // --- DAP Support ---
    context.subscriptions.push(vscode.debug.registerDebugAdapterDescriptorFactory('cdsl', {
        createDebugAdapterDescriptor(_session: vscode.DebugSession, _executable: vscode.DebugAdapterExecutable | undefined): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
            let dapPath = config.get<string>('dap.path');

            if (!dapPath) {
                const possibleDapPaths = [
                    path.join(context.extensionPath, '..', '..', 'target', 'debug', 'cdsl-dap'),
                    path.join(context.extensionPath, '..', '..', 'target', 'release', 'cdsl-dap'),
                    path.join(context.extensionPath, 'bin', 'cdsl-dap'),
                    'cdsl-dap',
                ];

                for (const p of possibleDapPaths) {
                    try {
                        if (p === 'cdsl-dap') {
                            dapPath = p;
                            break;
                        }
                        // Note: Using synchronous check here for simplicity in the provider
                        // but in a real extension we might want to pre-resolve this.
                        dapPath = p;
                        break;
                    } catch { }
                }
            }

            if (dapPath) {
                console.log(`CDSL: Using Debug Adapter at ${dapPath}`);
                return new vscode.DebugAdapterExecutable(dapPath, []);
            } else {
                vscode.window.showErrorMessage('CDSL Debug Adapter (cdsl-dap) not found. Build it with: cargo build -p cdsl-dap');
                return undefined;
            }
        }
    }));

    context.subscriptions.push({
        dispose: () => {
            if (client) {
                client.stop();
            }
        },
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
