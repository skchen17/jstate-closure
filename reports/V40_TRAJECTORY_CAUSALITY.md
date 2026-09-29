# V40 — Trajectory Causality

Status: **trajectory intervention not run**.

The planned primary test follows Qwen REC across early, middle, and late recurrent-block groups, replacing native donor states at selected checkpoints while all downstream computation runs naturally. Checkpoints must capture REC, Conv, hidden states, future logits, and externally defined task-variable recovery. No future output copying is allowed.

The precise block grouping, intervention schedule, and equality instrumentation must be sealed before formal outcomes. There is currently no V40 evidence that an early replacement alters how later states are used, or that a donor-like representation restores the task variable.
