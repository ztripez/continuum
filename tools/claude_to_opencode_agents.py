#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import sys

def split_frontmatter(text):
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return None, text
    front_lines = lines[1:end]
    body_lines = lines[end + 1 :]
    body = "\n".join(body_lines)
    if text.endswith("\n"):
        body += "\n"
    return front_lines, body

def parse_frontmatter(lines):
    data = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = ""
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            quote = value[0]
            inner = value[1:-1]
            if quote == '"':
                try:
                    inner = bytes(inner, "utf-8").decode("unicode_escape")
                except Exception:
                    pass
            else:
                inner = inner.replace("''", "'")
            data[key] = inner
        else:
            data[key] = value
    return data

def map_model(model, model_map, default_model, allow_unmapped):
    if model is None:
        if default_model:
            return default_model
        raise ValueError("missing model and no --default-model provided")
    if model in model_map:
        return model_map[model]
    if "/" in model:
        return model
    if default_model:
        return default_model
    if allow_unmapped:
        return model
    raise ValueError(f"unmapped model '{model}' (use --model-map or --default-model)")

def build_output(description, model, temperature, body):
    fm = ["---", f"description: {json.dumps(description)}", "mode: subagent", f"model: {model}"]
    if temperature is not None:
        fm.append(f"temperature: {temperature}")
    fm.append("---")
    content = "\n".join(fm) + "\n\n"
    if body:
        content += body.lstrip("\n")
    return content

def main():
    parser = argparse.ArgumentParser(description="Convert Claude agent markdown files to OpenCode subagent markdown files.")
    parser.add_argument("--src", default=".claude/agents", help="Source directory with Claude agent markdown files")
    parser.add_argument("--dst", default=".opencode/agent", help="Destination directory for OpenCode agent markdown files")
    parser.add_argument("--model-map", action="append", default=[], help="Model mapping in form claude=opencode")
    parser.add_argument("--default-model", help="Default OpenCode model to use when no mapping exists")
    parser.add_argument("--allow-unmapped", action="store_true", help="Allow unmapped models to pass through as-is")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination files if they exist")
    parser.add_argument("--dry-run", action="store_true", help="Print planned output paths without writing files")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    model_map = {}
    for item in args.model_map:
        if "=" not in item:
            parser.error(f"--model-map must be in form claude=opencode (got '{item}')")
        k, v = item.split("=", 1)
        model_map[k.strip()] = v.strip()

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"Source directory not found: {src_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in src_dir.rglob("*.md") if p.is_file())
    if not files:
        print(f"No markdown files found under {src_dir}", file=sys.stderr)
        return 1

    errors = []
    planned = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        front_lines, body = split_frontmatter(text)
        if front_lines is None:
            errors.append(f"{path}: missing YAML frontmatter")
            continue
        fm = parse_frontmatter(front_lines)
        description = fm.get("description", "")
        if not description:
            errors.append(f"{path}: missing description in frontmatter")
            continue
        temperature = fm.get("temperature")
        model = fm.get("model")
        try:
            model_out = map_model(model, model_map, args.default_model, args.allow_unmapped)
        except ValueError as e:
            errors.append(f"{path}: {e}")
            continue

        name = path.stem
        out_path = dst_dir / f"{name}.md"
        planned.append((path, out_path, description, model_out, temperature, body))

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 2

    if args.dry_run:
        for _, out_path, *_ in planned:
            print(out_path)
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    for _, out_path, description, model_out, temperature, body in planned:
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing: {out_path}")
            continue
        content = build_output(description, model_out, temperature, body)
        out_path.write_text(content, encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
