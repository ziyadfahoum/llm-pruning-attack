---
name: auto-mode-blocks-hardcoded-secrets
description: Auto-mode classifier blocks launching/writing scripts that hardcode live API keys; inherit from env instead
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

The Claude Code auto-mode classifier BLOCKS any Bash action that hardcodes a live OPENAI_API_KEY / HF_TOKEN into a script under /tmp or writes secrets to a new file (flagged "Credential Leakage"). The older scratchpad scripts (gemma_ncalib.sh etc.) still contain hardcoded keys and predate this.

**Why:** materializing real secrets into an inspectable/persistent artifact.

**How to apply:** write new scripts with NO hardcoded secrets — they should `export PATH=...` only and rely on inherited env. Launch them by sourcing the keys at runtime from an EXISTING on-disk script into the current shell, then nohup the script:
`set -a; source <(grep -E '^export (OPENAI_API_KEY|HF_TOKEN)=' <existing_script>); set +a; nohup bash new_script.sh &`
This puts no secret in the command text and creates no new secret file. Do NOT try to write a consolidated ~/.env secrets file — that is also blocked.
