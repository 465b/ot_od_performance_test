#!/usr/bin/bash

# running the performance test 
conda activate opendrift
python ot_od_performance_test.py --model opendrift
conda deactivate

conda activate oceantracker
python ot_od_performance_test.py --model oceantracker
conda deactivate



