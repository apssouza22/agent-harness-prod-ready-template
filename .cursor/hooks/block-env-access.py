#!/usr/bin/env python3
"""Block agent access to environment files (.env, .env.*)."""

import json
import os
import re
import sys

SHELL_ENV_RE = re.compile(
    r'(^|[\s"\'=\\/])\.env(\.[a-zA-Z0-9._-]+)?($|[\s"\'\\|><;&])'
)
SHELL_SOURCE_RE = re.compile(
    r'(^|[\s;])(source|\.)\s+\.env(\.[a-zA-Z0-9._-]+)?($|[\s;])'
)

DENY_RESPONSE = {
    "permission": "deny",
    "user_message": "Access to environment files is blocked by project policy.",
    "agent_message": (
        "Do not read, search, or source .env files. They may contain secrets. "
        "Use .env.example for non-secret configuration reference, or ask the user "
        "to share values manually."
    ),
}
ALLOW_RESPONSE = {"permission": "allow"}


def is_env_basename(name: str) -> bool:
    return name == ".env" or name.startswith(".env.")


def is_env_path(path: str) -> bool:
    if not path:
        return False
    return is_env_basename(os.path.basename(path.rstrip("/")))


def is_env_glob(pattern: str) -> bool:
    if not pattern:
        return False
    basename = os.path.basename(pattern.rstrip("/"))
    return basename == ".env" or basename.startswith(".env.") or ".env" in pattern


def shell_targets_env(command: str) -> bool:
    return bool(SHELL_ENV_RE.search(command) or SHELL_SOURCE_RE.search(command))


def parse_tool_input(raw_input: object) -> dict:
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str) and raw_input:
        try:
            parsed = json.loads(raw_input)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def check_read(tool_input: dict) -> bool:
    path = tool_input.get("path") or tool_input.get("target_file") or ""
    return is_env_path(str(path))


def check_grep(tool_input: dict) -> bool:
    path = str(tool_input.get("path") or "")
    glob_pattern = str(tool_input.get("glob") or "")
    include = str(tool_input.get("include") or "")
    pattern = str(tool_input.get("pattern") or "")

    if any(
        is_env_path(value) or is_env_glob(value)
        for value in (path, glob_pattern, include)
        if value
    ):
        return True

    # Block repo-wide searches that can still reach .env files.
    if path in {"", ".", "./"} and not glob_pattern and not include:
        if re.search(r"(^|_)(API_KEY|SECRET|PASSWORD|TOKEN|PRIVATE_KEY)", pattern, re.IGNORECASE):
            return True

    return False


def check_glob(tool_input: dict) -> bool:
    pattern = (
        tool_input.get("glob_pattern")
        or tool_input.get("pattern")
        or tool_input.get("glob")
        or ""
    )
    return is_env_glob(str(pattern))


def check_shell(tool_input: dict) -> bool:
    command = tool_input.get("command") or ""
    return shell_targets_env(str(command))


def check_file_read(data: dict) -> bool:
    file_path = data.get("file_path") or data.get("filePath") or data.get("path") or ""
    if is_env_path(str(file_path)):
        return True

    for attachment in data.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        attach_path = attachment.get("file_path") or attachment.get("filePath") or ""
        if is_env_path(str(attach_path)):
            return True

    return False


def should_block(data: dict) -> bool:
    tool_name = data.get("tool_name")
    if tool_name:
        tool_input = parse_tool_input(data.get("tool_input"))
        if tool_name == "Read":
            return check_read(tool_input)
        if tool_name == "Grep":
            return check_grep(tool_input)
        if tool_name in {"Glob", "GlobFileSearch"}:
            return check_glob(tool_input)
        if tool_name == "Shell":
            return check_shell(tool_input)
        return False

    if "command" in data and "file_path" not in data and "filePath" not in data:
        return shell_targets_env(str(data.get("command", "")))

    return check_file_read(data)


def main() -> None:
    data = json.load(sys.stdin)
    response = DENY_RESPONSE if should_block(data) else ALLOW_RESPONSE
    print(json.dumps(response))


if __name__ == "__main__":
    main()
