# V21 nested action-data scaling

S2 full 12-action state context, Z1 and G2 were held fixed. Only decoder-supervision action count changed, so this is not a joint test of how many actions suffice to *measure* S2.

| train decoder actions | unseen-direction L2 | median cosine | train-selected ridge |
|---:|---:|---:|---:|
| 4 | 0.9964 | 0.0959 | 1.00000 |
| 8 | 0.6980 | 0.3596 | 1.00000 |
| 12 | 0.6386 | 0.4403 | 0.00010 |

The 4→12 action L2 reduction is **0.3578**, with a Z1 validation train-span residual of **0.7437**. The frozen action-data-limited diagnostic is **True**. Counts 16/18 are not available within the sealed V20 12-train/6-validation split without reassigning validation labels, so they were not used as training actions.
