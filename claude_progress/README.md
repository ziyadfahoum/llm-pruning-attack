# claude_progress — Claude's memory notes for this project

These are the granular per-topic notes Claude wrote during the project (per-model results, gotchas,
decisions). `MEMORY.md` is the index. See `../HANDOFF.md` for the full consolidated state.

## To make Claude AUTO-LOAD these on a new machine
Claude reads memory from `~/.claude/projects/<dash-encoded-repo-path>/memory/`. Copy these there:
```bash
mkdir -p ~/.claude/projects/-home-<user>-llm-pruning-attack-alphaedit/memory
cp claude_progress/*.md ~/.claude/projects/-home-<user>-llm-pruning-attack-alphaedit/memory/
```
(Replace `-home-<user>-...` with your repo's absolute path, `/`→`-`.) Then Claude recalls them
automatically. Otherwise just tell Claude "read claude_progress/ and HANDOFF.md".
