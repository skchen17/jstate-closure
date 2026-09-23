# V38 — Q1 Control

Frozen 4/family development and 2/family validation control subsets compare natural Q234 versus Q1+Q234 (all donor REC on donor Conv):

|model|role|n|Q234 donor error|all Q donor error|paired Q1 error gain|Q1 vector increment|
|---|---|---|---|---|---|---|
|Q|development|20|11.202|9.165|1.393|5.871|
|Q|validation|10|8.943|7.733|1.495|5.607|
|F|development|20|8.567|7.707|0.454|3.906|
|F|validation|10|7.275|6.645|0.710|3.236|

Q1 is not dispensable on these subsets: adding it further reduces donor error. This is secondary and does not reopen layer search.
