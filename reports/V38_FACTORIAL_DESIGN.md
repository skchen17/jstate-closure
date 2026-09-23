# V38 — Factorial Design

Fixed background: donor Conv, recipient REC, recipient-native KV. Q2/Q3/Q4 are the 7–12/13–18/19–24 recurrent-layer groups. Conditions: `R000`, `R100`, `R010`, `R001`, `R110`, `R101`, `R011`, `R111`; group membership and six probes are in `execution_plan_v38.json` and model designs. All algebra uses state-wise six-probe vectors.

`Eabc=Yabc−Y000`; additive=`E100+E010+E001`; second-order=`E110+E101+E011−E100−E010−E001`; three-way=`E111−E110−E101−E011+E100+E010+E001`. The pasted expressions contained typographic `*` artifacts; the standard inclusion–exclusion formulas were frozen before outcomes.
