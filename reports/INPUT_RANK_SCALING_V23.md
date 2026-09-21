# Input-Rank Scaling — V23

| m | Gram rank | condition | JVP r90/r95/r99 | finite r90/r95/r99 |
|---:|---:|---:|---:|---:|
| 64 | 64 | 9.046e+04 | [11.0, 17.0, 31.5] | [15.5, 21.0, 33.0] |
| 128 | 128 | 7.690e+05 | [15.0, 23.0, 48.0] | [18.5, 27.0, 46.5] |
| 256 | 254 | 2.979e+06 | [17.5, 28.0, 65.0] | [23.0, 33.5, 60.5] |
| 512 | 417 | 5.413e+06 | [19.0, 31.0, 75.0] | [25.0, 36.5, 67.5] |

JVP operator states: 20; finite operator states: 10. The frozen last-two-doubling saturation criterion gives JVP=`FALSE`, finite=`FALSE`, joint=`FALSE`. Input and output energy ranks share the operator singular spectrum; their singular vectors remain distinct.
