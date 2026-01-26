# VSCode Extension Architecture Rules

## Fundamental Constraint: Pure Transport + UI Layer

The VSCode extension is a **transport and UI layer**. It exists ONLY to:
1. Provide VSCode language registration and UI integration
2. Launch and communicate with LSP/DAP processes
3. Present engine data in VSCode UI elements
4. Handle VSCode-specific commands

## Forbidden Patterns

### ❌ NO Custom Data Structures Over Engine Types

The extension must NOT:
- Define custom symbol types (use LSP's interpretation of engine types)
- Cache or duplicate engine state
- Parse CDSL files directly (LSP does this via engine)
- Implement language features (LSP does this via engine)

**Wrong:**
```typescript
// DON'T parse CDSL or define symbol types
interface CdslSymbol {
    kind: 'signal' | 'field' | 'operator';  // ❌
    name: string;
    type: string;
}

function parseCdsl(text: string): CdslSymbol[] {
    // ❌ Extension should never parse CDSL
}
```

**Right:**
```typescript
// Only launch LSP and let it talk to the engine
const client = new LanguageClient(
    'cdsl',
    'CDSL Language Server',
    serverOptions,
    clientOptions
);

client.start();  // LSP handles all language features
```

## Extension Responsibilities

### ✅ Language Registration

```typescript
// Register language and activate LSP
export function activate(context: vscode.ExtensionContext) {
    // 1. Register language
    context.subscriptions.push(
        vscode.languages.setLanguageConfiguration('cdsl', languageConfig)
    );
    
    // 2. Launch LSP client
    const client = createLanguageClient(context);
    context.subscriptions.push(client.start());
    
    // 3. Register extension commands (if needed)
    context.subscriptions.push(
        vscode.commands.registerCommand('cdsl.restart-lsp', () => {
            client.restart();
        })
    );
}
```

### ✅ LSP Client Configuration

```typescript
const serverOptions: ServerOptions = {
    command: 'cdsl-lsp',  // LSP binary path
    args: [],
};

const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'cdsl' }],
    synchronize: {
        fileEvents: vscode.workspace.createFileSystemWatcher('**/*.cdsl'),
    },
};
```

### ✅ DAP Configuration

```typescript
// Configure debug adapter
vscode.debug.registerDebugAdapterDescriptorFactory('cdsl', {
    createDebugAdapterDescriptor(session) {
        return new vscode.DebugAdapterExecutable('cdsl-dap');
    },
});

// Provide debug configuration
vscode.debug.registerDebugConfigurationProvider('cdsl', {
    resolveDebugConfiguration(folder, config) {
        // Minimal config - DAP handles the rest via engine
        return {
            type: 'cdsl',
            request: 'launch',
            name: config.name || 'CDSL Debug',
            world: config.world,  // Path to world
            scenario: config.scenario,
            ...config,
        };
    },
});
```

### ✅ UI Integration (Optional)

```typescript
// Custom tree views, webviews for engine data visualization
// BUT: Data comes from LSP queries, not parsed by extension

// Example: Signal inspector tree view
class SignalTreeProvider implements vscode.TreeDataProvider<SignalItem> {
    async getChildren(): Promise<SignalItem[]> {
        // Query LSP for symbol list
        const symbols = await client.sendRequest('workspace/symbol', {
            query: '',
        });
        
        // Filter to signals and display
        return symbols
            .filter(s => s.kind === vscode.SymbolKind.Variable)
            .map(s => new SignalItem(s.name, s.location));
    }
}
```

## File Structure

```
tools/vscode-cdsl/
├── package.json          # Extension manifest
├── src/
│   ├── extension.ts      # Activation, LSP/DAP launch
│   ├── lsp-client.ts     # LSP client configuration
│   ├── dap-config.ts     # DAP configuration
│   └── ui/               # Optional: custom views/commands
│       ├── signals.ts    # Signal inspector (queries LSP)
│       └── fields.ts     # Field visualization (queries LSP)
├── syntaxes/
│   └── cdsl.tmLanguage.json  # TextMate grammar (syntax highlighting)
└── language-configuration.json  # Brackets, comments, etc.
```

## Extension Scope

| Responsibility | Handled By | Extension Role |
|----------------|------------|----------------|
| Parse CDSL | Engine (via LSP) | None - just launch LSP |
| Symbol lookup | Engine (via LSP) | None - LSP provides |
| Type checking | Engine (via LSP) | None - LSP provides |
| Hover/completion | LSP | Forward VSCode requests |
| Diagnostics | Engine (via LSP) | Display LSP diagnostics |
| Debugging | Engine (via DAP) | Launch DAP, configure |
| Syntax highlighting | TextMate grammar | Provide grammar file |
| Custom UI | Extension | Query LSP for data |

## Custom Commands (If Needed)

```typescript
// Commands that don't fit LSP protocol
vscode.commands.registerCommand('cdsl.compile-world', async (uri: vscode.Uri) => {
    // Still delegate to LSP with custom request
    const result = await client.sendRequest('cdsl/compileWorld', {
        uri: uri.toString(),
    });
    
    vscode.window.showInformationMessage(
        `Compiled: ${result.nodeCount} nodes, ${result.errors.length} errors`
    );
});
```

## Rule Summary

1. **Launch, don't implement** - Extension launches LSP/DAP, doesn't parse or analyze
2. **Query, don't cache** - Get data from LSP on-demand, don't maintain state
3. **Display, don't interpret** - Show what LSP provides, don't add semantics
4. **Protocol types only** - Use LSP/DAP protocol types, not custom structures
5. **Engine is source of truth** - Extension → LSP → Engine (never bypass)

When in doubt: **If it's language features, LSP does it. If it's debugging, DAP does it. Extension just wires them to VSCode.**
