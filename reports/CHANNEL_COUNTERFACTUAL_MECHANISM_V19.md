# V19 channel × horizon causal mechanism

Channels refer to the requested and read-back persistent q support; labels were corrected append-only after inspecting frozen direction tensors. This is a new four-way estimand, not a reuse of V15–V18 channel rankings.

| q channel | h | states | rows | action match | median ‖N‖ | median ‖M‖ | median M/R |
|---|---|---|---|---|---|---|---|
| conv | h1 | 100 | 1840 | 1.0000 | 10.9000 | 5.8786 | 0.8542 |
| conv | h2 | 15 | 480 | 1.0000 | 4.5524 | 2.2571 | 0.8504 |
| conv | h4 | 15 | 480 | 1.0000 | 1.9128 | 1.1311 | 0.9011 |
| conv | h8 | 15 | 480 | 1.0000 | 0.9055 | 0.4633 | 1.0396 |
| joint | h1 | 100 | 1840 | 1.0000 | 1.6954 | 0.8072 | 0.2365 |
| joint | h2 | 15 | 480 | 1.0000 | 1.0897 | 0.4818 | 0.3897 |
| joint | h4 | 15 | 480 | 1.0000 | 1.1172 | 0.5694 | 0.5760 |
| joint | h8 | 15 | 480 | 1.0000 | 0.3226 | 0.3085 | 0.8866 |
| kv | h1 | 100 | 920 | 1.0000 | 0.1451 | 0.2191 | 0.0354 |
| kv | h2 | 15 | 240 | 1.0000 | 0.1398 | 0.2496 | 0.1261 |
| kv | h4 | 15 | 240 | 1.0000 | 0.1914 | 0.2898 | 0.2995 |
| kv | h8 | 15 | 240 | 1.0000 | 0.1726 | 0.2679 | 0.7587 |
| rec_conv | h1 | 100 | 920 | 1.0000 | 11.1543 | 6.6310 | 0.9599 |
| rec_conv | h2 | 15 | 240 | 1.0000 | 9.2694 | 2.2861 | 0.8994 |
| rec_conv | h4 | 15 | 240 | 1.0000 | 4.4890 | 1.3873 | 0.9750 |
| rec_conv | h8 | 15 | 240 | 1.0000 | 1.4276 | 0.4551 | 1.0790 |
| recurrent | h1 | 100 | 1840 | 1.0000 | 3.0812 | 0.7295 | 0.2884 |
| recurrent | h2 | 15 | 480 | 1.0000 | 2.3274 | 0.4420 | 0.4664 |
| recurrent | h4 | 15 | 480 | 1.0000 | 2.4224 | 0.5807 | 0.6209 |
| recurrent | h8 | 15 | 480 | 1.0000 | 0.6612 | 0.3394 | 0.9421 |

The REC+Conv and joint q families are distinct from isolated REC, Conv and KV; a weak natural KV effect is not evidence that KV is irrelevant in general.
