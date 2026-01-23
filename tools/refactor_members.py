#!/usr/bin/env python3
"""
Refactor CDSL member declarations to entity signals.

Converts:
    member entity.name.signal_name {
        ...
    }

To:
    entity entity.name {
        signal signal_name {
            ...
        }
    }
"""

import re
import sys
from pathlib import Path


def parse_entity_and_signal(member_decl: str) -> tuple[str, str]:
    """Parse 'member entity.name.signal_name' into ('entity.name', 'signal_name')"""
    match = re.match(r"member\s+([\w.]+)\.([\w]+)", member_decl)
    if not match:
        raise ValueError(f"Could not parse member declaration: {member_decl}")
    full_path = match.group(1)
    signal_name = match.group(2)
    return full_path, signal_name


def find_member_blocks(content: str) -> list[tuple[int, int, str, str]]:
    """
    Find all member blocks in content.
    Returns list of (start_line, end_line, entity_path, signal_name)
    """
    lines = content.splitlines()
    blocks = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for member declaration
        if line.strip().startswith("member "):
            # Parse entity and signal name
            try:
                entity_path, signal_name = parse_entity_and_signal(line.strip())
            except ValueError as e:
                print(f"Warning line {i + 1}: {e}", file=sys.stderr)
                i += 1
                continue

            # Find matching closing brace
            brace_count = 0
            start_line = i
            found_open = False

            for j in range(i, len(lines)):
                for char in lines[j]:
                    if char == "{":
                        brace_count += 1
                        found_open = True
                    elif char == "}":
                        brace_count -= 1

                        if found_open and brace_count == 0:
                            # Found the closing brace
                            blocks.append((start_line, j, entity_path, signal_name))
                            i = j + 1
                            break

                if found_open and brace_count == 0:
                    break
            else:
                print(
                    f"Warning: Unclosed member block at line {start_line + 1}",
                    file=sys.stderr,
                )
                i += 1
        else:
            i += 1

    return blocks


def extract_signal_body(lines: list[str], start_line: int, end_line: int) -> str:
    """Extract the body of a member block (everything between the braces)"""
    # Find the line with the opening brace
    first_line = lines[start_line]
    if "{" in first_line:
        # Opening brace on same line as member declaration
        brace_pos = first_line.index("{")
        body_lines = [first_line[brace_pos + 1 :].rstrip()]
        body_lines.extend(lines[start_line + 1 : end_line])
        body_lines.append(lines[end_line][: lines[end_line].rindex("}")])
    else:
        # Opening brace on next line
        body_lines = lines[start_line + 1 : end_line + 1]
        # Remove opening brace
        if body_lines and "{" in body_lines[0]:
            body_lines[0] = body_lines[0][body_lines[0].index("{") + 1 :].rstrip()
        # Remove closing brace
        if body_lines and "}" in body_lines[-1]:
            body_lines[-1] = body_lines[-1][: body_lines[-1].rindex("}")].rstrip()

    return "\n".join(body_lines).strip()


def convert_member_to_signal(content: str) -> str:
    """Convert all member blocks to signal blocks"""
    lines = content.splitlines()

    # Find all member blocks
    blocks = find_member_blocks(content)

    if not blocks:
        return content

    print(f"Found {len(blocks)} member blocks", file=sys.stderr)

    # Group by entity
    entity_signals = {}
    for start, end, entity_path, signal_name in blocks:
        if entity_path not in entity_signals:
            entity_signals[entity_path] = []

        # Extract signal body
        signal_body = extract_signal_body(lines, start, end)

        entity_signals[entity_path].append(
            {"name": signal_name, "body": signal_body, "start": start, "end": end}
        )

    # Find entity declarations and insert signals
    result_lines = []
    removed_lines = set()

    # Mark all member blocks for removal
    for start, end, _, _ in blocks:
        for i in range(start, end + 1):
            removed_lines.add(i)

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip removed member blocks
        if i in removed_lines:
            i += 1
            continue

        # Check if this is an entity declaration we need to modify
        entity_match = re.match(r"^entity\s+([\w.]+)\s*{", line.strip())
        if entity_match:
            entity_path = entity_match.group(1)

            if entity_path in entity_signals:
                # Found entity that needs signals added
                result_lines.append(lines[i])

                # Find the closing brace of the entity
                brace_count = 1
                entity_start = i
                entity_end = i

                for j in range(i + 1, len(lines)):
                    if j in removed_lines:
                        continue

                    for char in lines[j]:
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1

                            if brace_count == 0:
                                entity_end = j
                                break

                    if brace_count == 0:
                        break

                # Copy existing entity body
                for j in range(i + 1, entity_end):
                    if j not in removed_lines:
                        result_lines.append(lines[j])

                # Add signal blocks
                for sig in entity_signals[entity_path]:
                    result_lines.append("")
                    result_lines.append(f"    signal {sig['name']} {{")
                    # Indent signal body by 8 spaces (4 for entity, 4 for signal)
                    for body_line in sig["body"].splitlines():
                        if body_line.strip():
                            result_lines.append("    " + body_line)
                        else:
                            result_lines.append("")
                    result_lines.append("    }")

                # Add closing brace
                result_lines.append(lines[entity_end])

                i = entity_end + 1
                continue

        result_lines.append(lines[i])
        i += 1

    return "\n".join(result_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: refactor_members.py <file.cdsl>", file=sys.stderr)
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    content = file_path.read_text()
    converted = convert_member_to_signal(content)

    # Write result
    file_path.write_text(converted)
    print(f"Converted {file_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
