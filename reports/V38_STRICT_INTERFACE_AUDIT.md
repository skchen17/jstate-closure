# V38 — Strict Interface Audit

Exact native REC writeback, donor Conv preservation, recipient-native KV, instrumented bitwise replay and six-probe continuation passed fresh V38 calibration. On frozen control subsets, median donor errors:

|model|role|recipient|REC-only|Conv-only|joint|wrong token|shuffle*|signflip*|read ceiling|
|---|---|---|---|---|---|---|---|---|---|
|Q|development|50.413|50.347|16.343|9.165|14.978|111.079|24.390|10.744|
|Q|validation|53.036|48.865|15.917|7.733|14.929|107.942|26.292|8.598|
|F|development|52.228|51.244|13.404|7.707|14.235|85.482|21.117|8.056|
|F|validation|52.151|52.406|12.391|6.645|12.587|86.923|22.644|6.833|

`*` Off-manifold controls, not natural states. Wrong token is same-prefix natural alternative. Q234 read-interface output patch is explicitly an interface ceiling and never counted as a natural mechanism. Raw tensor arrays remain local/ignored by Git; table records, hashes and seals are versioned.
