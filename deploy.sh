source ~/miniconda3/etc/profile.d/conda.sh

# running the performance test 
conda activate oceantracker
python ot_od_performance_test.py --model oceantracker --dataset schism_small
python ot_od_performance_test.py --model oceantracker --dataset schism_large
python ot_od_performance_test.py --model oceantracker --dataset rom
conda deactivate

conda activate opendrift
python ot_od_performance_test.py --model opendrift --dataset schism_small
python ot_od_performance_test.py --model opendrift --dataset schism_large
python ot_od_performance_test.py --model opendrift --dataset rom
conda deactivate


conda activate oceantracker
python ot_od_performance_test.py --model oceantracker --dataset schism_small  --output=3600
python ot_od_performance_test.py --model oceantracker --dataset schism_large  --output=3600
python ot_od_performance_test.py --model oceantracker --dataset rom  --output=3600
conda deactivate

conda activate opendrift
python ot_od_performance_test.py --model opendrift --dataset schism_small  --output=3600
python ot_od_performance_test.py --model opendrift --dataset schism_large  --output=3600
python ot_od_performance_test.py --model opendrift --dataset rom  --output=3600
conda deactivate