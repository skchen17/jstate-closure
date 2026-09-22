# Natural Write Boundaries — V28

| field | incoming read in token t | natural outgoing update | current-token relation | valid post-forward interception |
|---|---|---|---|---|
| REC | GatedDeltaNet reads `recurrent_states` as initial state | gated delta rule computes and caches new recurrent state | mixer output is downstream of the recurrence; no claim that the update is current-invisible | exact old-state restoration |
| Conv | GatedDeltaNet reads `conv_states` for causal convolution | token update changes fixed-shape convolution context | current mixer consumes the convolution result | exact old-state restoration |
| KV | full attention reads old keys/values and includes token-t key/value | cache appends a new KV slot | current self-attention can consume the new slot | same-length new-slot replacement or same-length donor transplant |

Simply shortening KV to the previous cache length changes the next-token positional semantics. Therefore `KV_old_state_restore_valid = FALSE`; `KV_last_slot_previous_copy` is explicitly a native length-preserving diagnostic, **not** exact `P_out → P_in`.
