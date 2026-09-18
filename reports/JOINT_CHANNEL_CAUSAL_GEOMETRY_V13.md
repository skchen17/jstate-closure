# Joint REC / convolution / KV causal geometry — V13

- fraction of frozen probe directions with >10% score energy in at least two channels: `0.602`
- dominant directions judged cross-channel: `True`
- same-prompt tangent rotation by dominant channel: `{'conv': 21.926273423433305, 'kv': 36.89069223999977, 'recurrent': 25.403882221877573}`

This joint loading explains why independently factorized variance PCA can discard low-variance causal combinations spanning REC, convolution, and KV fields.
