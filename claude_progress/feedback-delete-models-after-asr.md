---
name: feedback-delete-models-after-asr
description: Always delete model checkpoints after calculating ASR unless user explicitly says to keep them
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Always delete model checkpoints (unpruned + pruned) after calculating ASR, UNLESS:
- The user explicitly says to keep them, OR
- The run achieves a new best gap (pruned ASR − unpruned ASR). Current best: 51.7%.

**Why:** Disk space is tight (~30-60G free) and each checkpoint is ~15G. Leaving them wastes space and can block subsequent runs. But the best model should be preserved for further analysis / the final paper.

**How to apply:** Add cleanup logic at end of run scripts — compare gap to best, delete if worse, keep if better.
