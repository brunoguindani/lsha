#!/bin/bash

num_experiments=5
base_seed=20250318

for ((i=0; i<num_experiments; i++)); do
  python scalability/scalability_events.py $((base_seed+i))
  python scalability/scalability_traces.py $((base_seed+i))
  base_seed=$((base_seed+10000))
done
