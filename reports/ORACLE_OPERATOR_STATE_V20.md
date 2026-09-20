# V20 oracle operator coordinate

`C_oracle(P)` is inferred from the same state's 12 positive train-action responses. Decoder fitting uses only train-state positive train-action responses and the continuous raw-direction descriptor (cosines to frozen train anchors plus channel norms). Validation directions, signs, amplitudes and action combinations do not supervise either coordinate or decoder. It is not an encoder from raw P.

Selected diagnostic model: `bilinear_latent_operator`, k=128; selected solely by validation unseen-direction stack relative L2. Seen-direction new-state L2=0.8236; unseen-direction L2=0.9273; unseen-sign train-direction L2=1.0056. Preliminary gate=False; full diagnostic gate=False. No compact operator-state is established; no inference about raw P compression follows.

Descriptor SHA `00311120a471c335335daac6d7ea5d23d491c187321a602d4a6bf2a51f453c97`; analysis freeze `c3f518b116a4bdfe54f7db38c3eddbc5c29519ea87b8a6881138a8f0fb7f5ed4`.
