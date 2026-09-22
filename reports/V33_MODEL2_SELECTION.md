# V33 Model-2 Selection

Independent pretrained hybrid family: `tiiuae/Falcon-H1-1.5B-Base`, fixed revision `06d7330266253c20784f58b0a846dc09a9d12cee`; separate from `Qwen/Qwen3.5-4B`. The official model card and config identify Falcon-H1 as a Transformer/Mamba-2 hybrid. The local checkpoint SHA-256 is `9acd2ef3e946e88a6e8cb14d38942f2ffbf6f52a5bc37312c1e0cc64574f1aeb`; tokenizer SHA-256 `eb7825ecac026cc37e37c03d7e8d06d1f85c7ab8bdefabe81fc1b40f0ed7929a`; config SHA-256 `7ae20392a453be5f1a59b39b1e1f4fc09522901eacd64492e649672594ea4c77`. The checkpoint lives outside Git at `/data/CSK/J-space-project/models/Falcon-H1-1.5B-Base-06d7330`.

Runtime: Transformers 5.12.1, PyTorch 2.10.0+cu128, bfloat16, CUDA:0, eager attention, unfused/naive Mamba path, context cap 512. Model identity and thresholds were frozen in `aa843d414c7ad5866a47beebc665db9d0379d2a622215b1465cdc4d6ae41b600` before model-2 causal response observation. This model is architecturally independent, but its Mamba-2 update algebra is not identical to Qwen3.5 gated delta recurrence.

Official model card: https://huggingface.co/tiiuae/Falcon-H1-1.5B-Base . Machine evidence: `results/v33/processed/`, immutable stage freezes in `artifacts/cross_model_rec_conv_v33*.freeze.json`.
