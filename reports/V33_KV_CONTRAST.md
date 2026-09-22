# V33 Attention-KV Contrast

On the frozen 20-state subset, KV-only new-slot donor replacement has median donor relative L2 0.984; CONV2+KV 0.189; REC2+CONV2 with recipient-native KV 0.157. Thus the tested immediate donor future is substantially reproduced through the recurrent/Conv route while KV-only is weak for this endpoint. Older KV history remains recipient-native in all partial contrasts; this does not deny general attention memory importance. Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
