# HTLM file for the instantiation of the experiment profile

description: GASP DIFF

keep_keys:
  - data.name
  - data.batch_size

model_repo: /Users/laurentvidal/Programs/deep_learning/inverse_material_design/GASP-DIFF

id_mode: sequential   # choose: sequential | hash | timestamp | uuid

# hash_length: 8

tags:
  - medium
  - HBTV

default_extra_files:
  - "logs/*.log"
  - "best*.txt"