@echo off

rem Running the performance test

call activate oceantracker
python ot_od_performance_test.py --model oceantracker
call deactivate

call activate opendrift
python ot_od_performance_test.py --model opendrift
call deactivate