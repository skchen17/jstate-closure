# Compact-Memory Trace Audit

Canonical records: 4000.
Representation screen authorized: True.
Duplicate IDs: 0; split overlaps: 0; invalid tensors: 0.

`teacher_correct` is ground-truth trajectory correctness; parseable-but-wrong traces remain available only for teacher-dynamics imitation.

|   attempted | family                      |   invalid_unparseable |   length |   parseable |   parseable_but_wrong | split      |   teacher_correct |   trace_valid |
|------------:|:----------------------------|----------------------:|---------:|------------:|----------------------:|:-----------|------------------:|--------------:|
|         100 | iterated_modular_arithmetic |                     0 |        8 |         100 |                    97 | test       |                 3 |           100 |
|         100 | iterated_modular_arithmetic |                     0 |       16 |         100 |                   100 | test       |                 0 |           100 |
|         100 | iterated_modular_arithmetic |                    44 |       32 |          56 |                    56 | test       |                 0 |           100 |
|         100 | synthetic_state_machine     |                    10 |        8 |          90 |                    90 | test       |                 0 |           100 |
|         100 | synthetic_state_machine     |                    50 |       16 |          50 |                    50 | test       |                 0 |           100 |
|         100 | synthetic_state_machine     |                    30 |       32 |          70 |                    69 | test       |                 1 |           100 |
|         467 | iterated_modular_arithmetic |                     0 |        8 |         467 |                   461 | train      |                 6 |           467 |
|         467 | iterated_modular_arithmetic |                     0 |       16 |         467 |                   467 | train      |                 0 |           467 |
|         466 | iterated_modular_arithmetic |                   164 |       32 |         302 |                   302 | train      |                 0 |           466 |
|         467 | synthetic_state_machine     |                    33 |        8 |         434 |                   434 | train      |                 0 |           467 |
|         467 | synthetic_state_machine     |                   225 |       16 |         242 |                   242 | train      |                 0 |           467 |
|         466 | synthetic_state_machine     |                   133 |       32 |         333 |                   333 | train      |                 0 |           466 |
|         100 | iterated_modular_arithmetic |                     0 |        8 |         100 |                    97 | validation |                 3 |           100 |
|         100 | iterated_modular_arithmetic |                     0 |       16 |         100 |                   100 | validation |                 0 |           100 |
|         100 | iterated_modular_arithmetic |                    34 |       32 |          66 |                    66 | validation |                 0 |           100 |
|         100 | synthetic_state_machine     |                     8 |        8 |          92 |                    92 | validation |                 0 |           100 |
|         100 | synthetic_state_machine     |                    50 |       16 |          50 |                    50 | validation |                 0 |           100 |
|         100 | synthetic_state_machine     |                    25 |       32 |          75 |                    75 | validation |                 0 |           100 |
