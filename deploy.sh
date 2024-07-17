#!/bin/bash

source /sw/buster-x64/anaconda3-2023.09/etc/profile.d/conda.sh
source ~/.bashrc


for i in {1..3}
do
    conda activate oceantracker_speedtest
    # python ot_od_performance_test.py --model oceantracker --dataset schism_small
    # python ot_od_performance_test.py --model oceantracker --dataset schism_large
    python ot_od_performance_test.py --model oceantracker --dataset rom
    python ot_od_performance_test.py --model oceantracker --dataset schism_estuary
    conda deactivate

    conda activate opendrift
    # python ot_od_performance_test.py --model opendrift --dataset schism_small
    # python ot_od_performance_test.py --model opendrift --dataset schism_large
    python ot_od_performance_test.py --model opendrift --dataset rom
    python ot_od_performance_test.py --model opendrift --dataset schism_estuary
    conda deactivate

    conda activate parcels
    # python ot_od_performance_test.py --model opendrift --dataset schism_small
    # python ot_od_performance_test.py --model opendrift --dataset schism_large
    python ot_od_performance_test.py --model parcels --dataset rom
    conda deactivate
done
