//! CDSL code formatter.
//!
//! Provides opinionated formatting for Continuum DSL files.

/// Format CDSL source code.
///
/// This is a simple line-based formatter that:
/// - Uses 4-space indentation
/// - Adds proper spacing around operators
/// - Ensures consistent blank lines between sections
/// - Trims trailing whitespace
pub fn format(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut indent_level: i32 = 0;
    let mut prev_line_empty = false;
    let mut prev_line_was_block_start = false;

    for line in source.lines() {
        let trimmed = line.trim();

        // Skip multiple consecutive empty lines
        if trimmed.is_empty() {
            if !prev_line_empty && !prev_line_was_block_start {
                output.push('\n');
            }
            prev_line_empty = true;
            prev_line_was_block_start = false;
            continue;
        }
        prev_line_empty = false;

        // Adjust indent for closing braces before writing
        let close_count = count_leading_closes(trimmed);
        indent_level -= close_count;
        if indent_level < 0 {
            indent_level = 0;
        }

        // Write indentation
        for _ in 0..indent_level {
            output.push_str("    ");
        }

        // Format the line content
        let formatted = format_line(trimmed);
        output.push_str(&formatted);
        output.push('\n');

        // Adjust indent for opening braces after writing
        let opens = trimmed.chars().filter(|c| *c == '{').count() as i32;
        let closes = trimmed.chars().filter(|c| *c == '}').count() as i32;
        indent_level += opens - closes + close_count;

        prev_line_was_block_start = trimmed.ends_with('{');
    }

    // Ensure file ends with single newline
    while output.ends_with("\n\n") {
        output.pop();
    }
    if !output.ends_with('\n') && !output.is_empty() {
        output.push('\n');
    }

    output
}

/// Count leading close braces for dedent calculation.
fn count_leading_closes(line: &str) -> i32 {
    let mut count = 0;
    for c in line.chars() {
        if c == '}' {
            count += 1;
        } else if !c.is_whitespace() {
            break;
        }
    }
    count
}

/// Format a single line of code.
fn format_line(line: &str) -> String {
    // Handle comments - preserve them as-is
    if line.starts_with('#') || line.starts_with("//") {
        return line.to_string();
    }

    // Handle block comments
    if line.starts_with("/*") || line.starts_with("*/") || line.starts_with('*') {
        return line.to_string();
    }

    let mut result = String::with_capacity(line.len() + 16);
    let mut chars = line.chars().peekable();
    let mut in_string = false;
    let mut in_unit = false;

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                in_string = !in_string;
                result.push(c);
            }
            '<' if !in_string => {
                // Check if this is a unit annotation or comparison
                // Units follow numbers or type names and contain no spaces
                let is_unit = is_unit_start(&result);
                if is_unit {
                    in_unit = true;
                    result.push(c);
                } else {
                    // Comparison operator
                    ensure_space_before(&mut result);
                    result.push(c);
                    if chars.peek() == Some(&'=') {
                        result.push(chars.next().unwrap());
                    } else if chars.peek() == Some(&'-') {
                        result.push(chars.next().unwrap());
                    }
                    ensure_space_after(&mut result, &mut chars);
                }
            }
            '>' if !in_string => {
                if in_unit {
                    in_unit = false;
                    result.push(c);
                } else {
                    // Comparison operator
                    ensure_space_before(&mut result);
                    result.push(c);
                    if chars.peek() == Some(&'=') {
                        result.push(chars.next().unwrap());
                    }
                    ensure_space_after(&mut result, &mut chars);
                }
            }
            '=' if !in_string => {
                if chars.peek() == Some(&'=') {
                    ensure_space_before(&mut result);
                    result.push(c);
                    result.push(chars.next().unwrap());
                    ensure_space_after(&mut result, &mut chars);
                } else {
                    // Assignment
                    ensure_space_before(&mut result);
                    result.push(c);
                    ensure_space_after(&mut result, &mut chars);
                }
            }
            '!' if !in_string && chars.peek() == Some(&'=') => {
                ensure_space_before(&mut result);
                result.push(c);
                result.push(chars.next().unwrap());
                ensure_space_after(&mut result, &mut chars);
            }
            '+' | '-' | '*' | '/' if !in_string && !in_unit => {
                // Check if this is a unary operator (after opening paren, comma, or operator)
                let is_unary = is_unary_context(&result);
                if is_unary {
                    result.push(c);
                } else {
                    ensure_space_before(&mut result);
                    result.push(c);
                    ensure_space_after(&mut result, &mut chars);
                }
            }
            '&' if !in_string && chars.peek() == Some(&'&') => {
                ensure_space_before(&mut result);
                result.push(c);
                result.push(chars.next().unwrap());
                ensure_space_after(&mut result, &mut chars);
            }
            '|' if !in_string && chars.peek() == Some(&'|') => {
                ensure_space_before(&mut result);
                result.push(c);
                result.push(chars.next().unwrap());
                ensure_space_after(&mut result, &mut chars);
            }
            ':' if !in_string => {
                // Colon in type annotations or key-value pairs
                result.push(c);
                // Add space after if not already present and next char isn't special
                if chars.peek().is_some_and(|&nc| !nc.is_whitespace() && nc != ':') {
                    result.push(' ');
                }
            }
            ',' if !in_string => {
                result.push(c);
                if chars.peek().is_some_and(|&nc| !nc.is_whitespace()) {
                    result.push(' ');
                }
            }
            '{' if !in_string => {
                ensure_space_before(&mut result);
                result.push(c);
            }
            '}' if !in_string => {
                result.push(c);
            }
            ' ' | '\t' if !in_string => {
                // Collapse multiple spaces
                if !result.ends_with(' ') && !result.is_empty() {
                    result.push(' ');
                }
            }
            _ => {
                result.push(c);
            }
        }
    }

    result.trim_end().to_string()
}

/// Check if we're at a position where `<` starts a unit/type annotation.
fn is_unit_start(preceding: &str) -> bool {
    let trimmed = preceding.trim_end();
    // Unit/type param follows a number, closing paren, or type name
    if trimmed.is_empty() {
        return false;
    }

    let last_char = trimmed.chars().last().unwrap();
    // After a digit or closing bracket of type parameters
    if last_char.is_ascii_digit() || last_char == ')' || last_char == '>' {
        return true;
    }

    // Check if preceding text ends with a type name (Scalar, Vector, Tensor, Map, etc.)
    let type_names = ["Scalar", "Vector", "Tensor", "Map"];
    for type_name in type_names {
        if trimmed.ends_with(type_name) {
            return true;
        }
    }

    // Also check for generic identifier ending (could be a custom type)
    // If it ends with an identifier character, check if it looks like a type
    if last_char.is_ascii_alphabetic() {
        // Get the last word
        let last_word: String = trimmed
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        // Types typically start with uppercase
        if last_word.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            return true;
        }
    }

    false
}

/// Check if we're in a unary operator context.
fn is_unary_context(preceding: &str) -> bool {
    let trimmed = preceding.trim_end();
    if trimmed.is_empty() {
        return true;
    }

    let last_char = trimmed.chars().last().unwrap();
    matches!(last_char, '(' | '[' | '{' | ',' | '=' | '<' | '>' | '+' | '-' | '*' | '/' | ':')
}

/// Ensure there's a space before the current position.
fn ensure_space_before(result: &mut String) {
    if !result.ends_with(' ') && !result.is_empty() {
        result.push(' ');
    }
}

/// Ensure there's a space after (consume any existing whitespace).
fn ensure_space_after(result: &mut String, chars: &mut std::iter::Peekable<std::str::Chars<'_>>) {
    while chars.peek().is_some_and(|&c| c == ' ' || c == '\t') {
        chars.next();
    }
    if chars.peek().is_some() {
        result.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_indentation() {
        let input = "signal.test {\n: Scalar<K>\nresolve {\nprev + 1.0\n}\n}";
        let expected = "signal.test {\n    : Scalar<K>\n    resolve {\n        prev + 1.0\n    }\n}\n";
        assert_eq!(format(input), expected);
    }

    #[test]
    fn test_operator_spacing() {
        let input = "x+y*z";
        let formatted = format_line(input);
        assert_eq!(formatted, "x + y * z");
    }

    #[test]
    fn test_comparison_spacing() {
        let input = "x<y&&z>0";
        let formatted = format_line(input);
        assert_eq!(formatted, "x < y && z > 0");
    }

    #[test]
    fn test_unit_preserved() {
        let input = "270.0<K>";
        let formatted = format_line(input);
        assert_eq!(formatted, "270.0<K>");
    }

    #[test]
    fn test_comments_preserved() {
        let input = "# This is a comment";
        let formatted = format_line(input);
        assert_eq!(formatted, input);
    }
}
