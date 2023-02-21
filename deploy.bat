@echo off

rem Running the performance test

call activate oceantracker
python ot_od_performance_test.py --model oceantracker --dataset schism_small
python ot_od_performance_test.py --model oceantracker --dataset schism_large
python ot_od_performance_test.py --model oceantracker --dataset rom
call deactivate

call activate opendrift
python ot_od_performance_test.py --model opendrift --dataset schism_small
python ot_od_performance_test.py --model opendrift --dataset schism_large
python ot_od_performance_test.py --model opendrift --dataset rom
call deactivate