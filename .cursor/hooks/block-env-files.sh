#!/usr/bin/env bash
# Block agent and Tab from reading environment files (.env, .env.*).
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | jq -r '.file_path // empty')

if [[ -z "$file_path" ]]; then
  echo '{"permission": "allow"}'
  exit 0
fi

basename="${file_path##*/}"

is_env_file=false
if [[ "$basename" == ".env" ]] || [[ "$basename" =~ ^\.env\. ]]; then
  is_env_file=true
fi

if [[ "$is_env_file" == true ]]; then
  jq -n \
    --arg path "$file_path" \
    --arg name "$basename" \
    '{
      "permission": "deny",
      "user_message": ("Reading environment files is blocked by project policy: " + $name)
    }'
  exit 0
fi

echo '{"permission": "allow"}'
exit 0
