description: |
  Trainings of GASP [REF] using the discrete graph diffusion method.

keep_keys:
  - data.name
  - data.batch_size

id_mode: hash   # choose: sequential | hash | timestamp | uuid
hash_length: 8
include_meta: true
tags:
  - gasp-diff
  - QM9
  - small

extra_files:
  - "logs/*.log"
  - "best*.txt"
