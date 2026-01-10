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

    // Find the LSP server executable
    let serverPath = config.get<string>('server.path');

    if (!serverPath) {
        // Try to find in common locations
        const possiblePaths = [
            // Development: cargo target directory (relative to workspace)
            path.join(context.extensionPath, '..', '..', 'target', 'debug', 'cdsl-lsp'),
            path.join(context.extensionPath, '..', '..', 'target', 'release', 'cdsl-lsp'),
            // Bundled with extension
            path.join(context.extensionPath, 'bin', 'cdsl-lsp'),
            // In PATH (just use the name)
            'cdsl-lsp',
        ];

        for (const p of possiblePaths) {
            try {
                // For 'cdsl-lsp' in PATH, we just try to use it
                if (p === 'cdsl-lsp') {
                    serverPath = p;
                    break;
                }
                // For absolute paths, check if file exists
                await vscode.workspace.fs.stat(vscode.Uri.file(p));
                serverPath = p;
                break;
            } catch {
                // File doesn't exist, try next
            }
        }
    }

    if (!serverPath) {
        vscode.window.showWarningMessage(
            'CDSL language server not found. Syntax highlighting will work, but diagnostics and completion are disabled. ' +
            'Build the server with: cargo build -p cdsl-lsp'
        );
        return;
    }

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

    client = new LanguageClient(
        'cdsl',
        'CDSL Language Server',
        serverOptions,
        clientOptions
    );

    await client.start();

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
