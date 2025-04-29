#!/bin/bash
numexp=$1
python fuzzing.py $numexp
python search.py $numexp
# python results.py
