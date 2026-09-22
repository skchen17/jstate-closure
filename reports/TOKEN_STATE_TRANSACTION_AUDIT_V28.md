# Token State Transaction Audit — V28

The Qwen3.5 hybrid text model has six linear-attention layers `[24, 25, 26, 28, 29, 30]` with REC and Conv state and two full-attention layers `[27, 31]` with KV state. The incoming cache after the prompt prefix is `P_t^in`; the model naturally computes the final prompt token and returns current layer-30 readout plus `P_t^out`. The next token consumes this returned cache.

The operative interception boundary is **after the complete token-t forward, before token t+1**. At this boundary natural outgoing fields already exist, current J/logits/semantic/workspace are captured and cannot be changed by later cache edits, and no current residual or output is overwritten. This is a temporal separation of an already-computed readout from a committed future condition; it is not proof of two independently implemented internal pathways.

The pilot found a historical naming ambiguity: layer-23 state-J is upstream of the tested writes, whereas the V26 h1 readout-J is layer 30. The formal V28 endpoint uses layer 30 at both t and t+1. The pilot development state `v13-train-4e521fb8e01b9354357f` was excluded from formal development; thresholds were unchanged. The amendment records that one pilot future response preceded endpoint freeze.
