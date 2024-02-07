# for cawthron hpc
# source ~/miniconda3/etc/profile.d/conda.sh

# for imf desktop
# source /sw/buster-x64/anaconda3-2022.05/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

# conda init bash

# running the performance test 
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