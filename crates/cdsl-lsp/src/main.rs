//! CDSL Language Server
//!
//! A Language Server Protocol implementation for the Continuum DSL.
//! Provides diagnostics, go-to-definition, and completion support.

use dashmap::DashMap;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

use continuum_dsl::{parse, CompilationUnit};

/// The CDSL language server backend.
struct Backend {
    /// LSP client for sending notifications.
    client: Client,
    /// Cached document contents.
    documents: DashMap<Url, String>,
    /// Cached ASTs for each document.
    asts: DashMap<Url, Option<CompilationUnit>>,
}

impl Backend {
    /// Parse a document and publish diagnostics.
    async fn parse_and_publish_diagnostics(&self, uri: Url, text: &str) {
        let (ast, errors) = parse(text);

        // Convert parse errors to LSP diagnostics
        let diagnostics: Vec<Diagnostic> = errors
            .iter()
            .map(|err| {
                let span = err.span();
                let (start_line, start_char) = offset_to_position(text, span.start);
                let (end_line, end_char) = offset_to_position(text, span.end);

                Diagnostic {
                    range: Range {
                        start: Position::new(start_line, start_char),
                        end: Position::new(end_line, end_char),
                    },
                    severity: Some(DiagnosticSeverity::ERROR),
                    source: Some("cdsl".to_string()),
                    message: format!("{}", err.reason()),
                    ..Default::default()
                }
            })
            .collect();

        // Cache the AST
        self.asts.insert(uri.clone(), ast);

        // Publish diagnostics
        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

/// Convert a byte offset to line/column position.
fn offset_to_position(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0u32;
    let mut col = 0u32;
    let mut current_offset = 0;

    for ch in text.chars() {
        if current_offset >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
        current_offset += ch.len_utf8();
    }

    (line, col)
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: Some(vec![".".to_string()]),
                    ..Default::default()
                }),
                definition_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                ..Default::default()
            },
            server_info: Some(ServerInfo {
                name: "cdsl-lsp".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "CDSL language server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;

        self.documents.insert(uri.clone(), text.clone());
        self.parse_and_publish_diagnostics(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;

        // We use FULL sync, so there's only one change with the full content
        if let Some(change) = params.content_changes.into_iter().next() {
            self.documents.insert(uri.clone(), change.text.clone());
            self.parse_and_publish_diagnostics(uri, &change.text).await;
        }
    }

    async fn did_close(&self, params: DidCloseTextDocumentParams) {
        let uri = params.text_document.uri;
        self.documents.remove(&uri);
        self.asts.remove(&uri);

        // Clear diagnostics
        self.client.publish_diagnostics(uri, vec![], None).await;
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = &params.text_document_position.text_document.uri;
        let _position = params.text_document_position.position;

        // Check if we have the document
        let _doc = match self.documents.get(uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };

        // Basic completion items for CDSL keywords
        let items = vec![
            // Top-level declarations
            completion_item("signal", CompletionItemKind::KEYWORD, "Signal declaration"),
            completion_item("field", CompletionItemKind::KEYWORD, "Field declaration"),
            completion_item("fracture", CompletionItemKind::KEYWORD, "Fracture declaration"),
            completion_item("impulse", CompletionItemKind::KEYWORD, "Impulse declaration"),
            completion_item("chronicle", CompletionItemKind::KEYWORD, "Chronicle declaration"),
            completion_item("entity", CompletionItemKind::KEYWORD, "Entity declaration"),
            completion_item("operator", CompletionItemKind::KEYWORD, "Operator declaration"),
            completion_item("strata", CompletionItemKind::KEYWORD, "Strata declaration"),
            completion_item("era", CompletionItemKind::KEYWORD, "Era declaration"),
            completion_item("const", CompletionItemKind::KEYWORD, "Const block"),
            completion_item("config", CompletionItemKind::KEYWORD, "Config block"),
            completion_item("fn", CompletionItemKind::KEYWORD, "Function definition"),
            completion_item("type", CompletionItemKind::KEYWORD, "Type definition"),
            // Block keywords
            completion_item("resolve", CompletionItemKind::KEYWORD, "Resolve block"),
            completion_item("measure", CompletionItemKind::KEYWORD, "Measure block"),
            completion_item("when", CompletionItemKind::KEYWORD, "When condition"),
            completion_item("emit", CompletionItemKind::KEYWORD, "Emit block"),
            completion_item("assert", CompletionItemKind::KEYWORD, "Assert block"),
            // Expression keywords
            completion_item("if", CompletionItemKind::KEYWORD, "Conditional expression"),
            completion_item("else", CompletionItemKind::KEYWORD, "Else branch"),
            completion_item("let", CompletionItemKind::KEYWORD, "Let binding"),
            completion_item("in", CompletionItemKind::KEYWORD, "In expression"),
            completion_item("prev", CompletionItemKind::VARIABLE, "Previous value"),
            completion_item("dt_raw", CompletionItemKind::VARIABLE, "Raw time delta"),
            completion_item("collected", CompletionItemKind::VARIABLE, "Collected value"),
            completion_item("payload", CompletionItemKind::VARIABLE, "Impulse payload"),
            // Types
            completion_item("Scalar", CompletionItemKind::TYPE_PARAMETER, "Scalar type"),
            completion_item("Vector", CompletionItemKind::TYPE_PARAMETER, "Vector type"),
            completion_item("Tensor", CompletionItemKind::TYPE_PARAMETER, "Tensor type"),
            // Built-in functions
            completion_item("clamp", CompletionItemKind::FUNCTION, "Clamp value to range"),
            completion_item("min", CompletionItemKind::FUNCTION, "Minimum of values"),
            completion_item("max", CompletionItemKind::FUNCTION, "Maximum of values"),
            completion_item("abs", CompletionItemKind::FUNCTION, "Absolute value"),
            completion_item("exp", CompletionItemKind::FUNCTION, "Exponential"),
            completion_item("ln", CompletionItemKind::FUNCTION, "Natural logarithm"),
            completion_item("log", CompletionItemKind::FUNCTION, "Logarithm"),
            completion_item("sqrt", CompletionItemKind::FUNCTION, "Square root"),
            completion_item("sin", CompletionItemKind::FUNCTION, "Sine"),
            completion_item("cos", CompletionItemKind::FUNCTION, "Cosine"),
            completion_item("tan", CompletionItemKind::FUNCTION, "Tangent"),
        ];

        Ok(Some(CompletionResponse::Array(items)))
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let _uri = &params.text_document_position_params.text_document.uri;
        let _position = params.text_document_position_params.position;

        // TODO: Implement go-to-definition using AST spans
        // This requires building a symbol table from the AST

        Ok(None)
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let _uri = &params.text_document_position_params.text_document.uri;
        let _position = params.text_document_position_params.position;

        // TODO: Implement hover using AST analysis

        Ok(None)
    }
}

fn completion_item(label: &str, kind: CompletionItemKind, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        documents: DashMap::new(),
        asts: DashMap::new(),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}
