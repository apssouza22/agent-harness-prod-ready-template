#!/usr/bin/env bash
# Block shell commands that read or source environment files (.env, .env.*).
set -euo pipefail

input=$(cat)
command=$(echo "$input" | jq -r '.command // empty')

if [[ -z "$command" ]]; then
  echo '{"permission": "allow"}'
  exit 0
fi

env_file_pattern='(^|[[:space:]"'"'"'=\\|/])\.env(\.[a-zA-Z0-9._-]+)?($|[[:space:]"'"'"'\\|><;&])'
source_pattern='(^|[[:space:];])(source|\.)[[:space:]]+\.env(\.[a-zA-Z0-9._-]+)?($|[[:space:];])'

is_env_access=false
if [[ "$command" =~ $env_file_pattern ]] || [[ "$command" =~ $source_pattern ]]; then
  is_env_access=true
fi

if [[ "$is_env_access" == true ]]; then
  jq -n \
    '{
      "permission": "deny",
      "user_message": "Shell access to environment files is blocked by project policy.",
      "agent_message": "Do not read or source .env files via shell. Use documented environment variables or ask the user to provide non-secret configuration details."
    }'
  exit 0
fi

echo '{"permission": "allow"}'
exit 0
