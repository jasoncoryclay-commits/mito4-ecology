#!/usr/bin/env bash
# setup_key.sh — save your OpenAI key ONCE on the pod's network drive, auto-load forever.
#
# Usage (interactive, key never appears in shell history):
#     bash setup_key.sh
# then paste your key at the prompt.
#
# After running once, every new shell auto-loads OPENAI_API_KEY. You never re-enter it.
set -euo pipefail

KEY_FILE="${KEY_FILE:-/workspace/.openai_key}"   # /workspace = persistent network drive

echo "This saves your OpenAI API key to ${KEY_FILE} and auto-loads it in every shell."
echo "Paste your key (input hidden), then press Enter:"
read -rs KEY
echo

# strip whitespace/newlines
KEY="$(printf '%s' "$KEY" | tr -d '[:space:]')"
if [ -z "$KEY" ]; then echo "No key entered. Aborting."; exit 1; fi
case "$KEY" in
  sk-*) : ;;                       # looks like a real key
  *) echo "Warning: key doesn't start with 'sk-'. Saving anyway." ;;
esac

printf '%s\n' "$KEY" > "$KEY_FILE"
chmod 600 "$KEY_FILE"
echo "saved -> ${KEY_FILE} (permissions 600)"

# auto-load line for future shells (idempotent)
LINE="export OPENAI_API_KEY=\$(cat ${KEY_FILE} | tr -d '[:space:]')"
if ! grep -qF "$KEY_FILE" ~/.bashrc 2>/dev/null; then
  printf '%s\n' "$LINE" >> ~/.bashrc
  echo "added auto-load to ~/.bashrc"
else
  echo "auto-load already in ~/.bashrc"
fi

# load into the CURRENT shell too
export OPENAI_API_KEY="$KEY"
echo "loaded into current shell. Check: ${OPENAI_API_KEY:0:8} (should be sk-proj- or sk-)"
echo
echo "Done. From now on just run:  python3 trustworthy_test.py"
echo "(new terminals auto-load the key; no need to re-enter it ever again.)"
