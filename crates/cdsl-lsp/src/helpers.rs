//! LSP protocol conversion helpers.
//!
//! Utility functions for converting between engine types and LSP protocol types.
//! These are pure transport-layer conversions with no domain logic.

use continuum_cdsl::ast::{Index as NodeIndex, Node, RoleId};
use tower_lsp::lsp_types::*;

/// Helper to create completion items.
pub fn completion_item(label: &str, kind: CompletionItemKind, detail: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail: Some(detail.to_string()),
        ..Default::default()
    }
}

/// Convert engine RoleId to LSP SymbolKind for protocol transport.
pub fn role_to_lsp_symbol_kind(role: RoleId) -> SymbolKind {
    match role {
        RoleId::Signal => SymbolKind::VARIABLE,
        RoleId::Field => SymbolKind::PROPERTY,
        RoleId::Operator => SymbolKind::METHOD,
        RoleId::Impulse => SymbolKind::EVENT,
        RoleId::Fracture => SymbolKind::EVENT,
        RoleId::Chronicle => SymbolKind::CLASS,
    }
}

/// Format hover markdown from an engine Node.
pub fn format_hover_from_node<I: NodeIndex>(node: &Node<I>) -> Hover {
    let mut parts = Vec::new();

    // Title (if present)
    if let Some(ref title) = node.title {
        parts.push(format!("**{}**", title));
    }

    // Role and path
    let role_name = node.role_id().spec().name;
    parts.push(format!("{}: `{}`", role_name, node.path));

    // Type (if resolved)
    if let Some(ref output) = node.output {
        parts.push(format!("Type: `{:?}`", output)); // TODO: proper Type formatting
    }

    // Documentation
    if let Some(ref doc) = node.doc {
        parts.push(doc.clone());
    }

    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: parts.join("\n\n"),
        }),
        range: None,
    }
}

/// Convert a byte offset in text to LSP line/character position.
pub fn offset_to_position(text: &str, offset: usize) -> (u32, u32) {
    let mut line = 0;
    let mut col = 0;
    for (i, c) in text.chars().enumerate() {
        if i == offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Convert an LSP line/character position to a byte offset in text.
pub fn position_to_offset(text: &str, pos: Position) -> usize {
    let mut line = 0;
    let mut col = 0;
    for (i, c) in text.chars().enumerate() {
        if line == pos.line && col == pos.character {
            return i;
        }
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    text.len()
}
