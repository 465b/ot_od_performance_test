#!/bin/bash

# for imf desktop
# source /sw/buster-x64/anaconda3-2022.05/etc/profile.d/conda.sh
# source ~/miniconda/etc/profile.d/conda.sh
# source /sw/buster-x64/anaconda3-2023.09/bin/conda
# source ~/.bashrc

# running the performance test 
# conda activate oceantracker_speedtest
# python ot_od_performance_test.py --model oceantracker --dataset schism_small
# python ot_od_performance_test.py --model oceantracker --dataset schism_large
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset rom
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset rom
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset rom
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset schism_estuary
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset schism_estuary
/home/zmaw/u301513/.conda/envs/oceantracker_speedtest/bin/python ot_od_performance_test.py --model oceantracker --dataset schism_estuary
# conda deactivate

# conda activate opendrift
# python ot_od_performance_test.py --model opendrift --dataset schism_small
# python ot_od_performance_test.py --model opendrift --dataset schism_large
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset rom
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset rom
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset rom
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset schism_estuary
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset schism_estuary
/home/zmaw/u301513/.conda/envs/opendrift/bin/python ot_od_performance_test.py --model opendrift --dataset schism_estuary
# conda deactivate