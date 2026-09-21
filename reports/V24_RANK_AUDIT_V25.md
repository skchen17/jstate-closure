# V24 Rank Audit — V25

The V24 raw, uncentered result is exactly reproduced: r95 is `1` at all 32 layers and raw top-1 energy fraction ranges from `0.999086` to `0.999450`. After subtracting the intervention-row mean independently within each frozen state, broad r95 is `13` at most layers and peaks at `14`; centered top-1 fraction is `0.403376`–`0.426321`. Row normalization without centering remains rank 1, while centered plus normalized r95 is `24`–`25`.

Therefore `LOW_RANK_RESPONSE_GEOMETRY_RECONFIRMED = FALSE` and `V24_RANK1_STATUS = MEASUREMENT_DEFINITION_SPECIFIC_UNCENTERED_EFFECT`. The rank-one statement is specific to an uncentered common-effect component, not a general low-dimensional intervention geometry claim.

Rows hash `bac7d41abf1c30d2523467b77a4c6f2dd2cf90cb99a68b5774ae0f9e25aa2f77`; medians hash `2c95ea7a2446b9af05b3ff5d52571d44c70ea8cc34704b2608e709f222323841`; spectra hash `74581ae47e67aa74237ad57d3822e060f76c777d7708576bb8f3652c99168126`.
