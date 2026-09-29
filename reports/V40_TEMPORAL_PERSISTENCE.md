# V40 — Temporal Persistence

Status: **single-state calibration pilots at selected delays; formal delay experiment not run**.

The frozen delay grid is 0, 1, 2, 4, 8, 16, and 32 shared neutral continuation tokens after the natural fork. ` neutral` passed tokenizer-level single-token checks in both models. One calibration state per model was run at delays 0, 1, and 32 under the sequential native-cache path; those pilots are not a population temporal profile. REC, Conv, KV, and REC+Conv effects and task-variable recovery remain to be measured formally at every delay.

No decay curve or channel half-life has been estimated. The analysis must first report empirical effect-versus-delay by task family; it must not impose exponential decay or infer a universal channel lifetime.
