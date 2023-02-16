#%%
import argparse
import pyproj as proj
import numpy as np
import os

'''
Here we compare the speed of OpenDrift with the speed of oceantracker03.
We test it on a SCHISM (unstruct) hindcast in Marlborough Sounds.
'''




#%%
# Preparation (def. params, model-set, etc)
# =========================================
#%%
# Parameterisation (for both models)
# ----------------------------------

cmd_parser = argparse.ArgumentParser(description='Run a test of the performance of OpenDrift and OceanTracker.')
cmd_parser.add_argument('--model', type=str, default='oceantracker', help='Which model to run. Options are "opendrift" and "oceantracker".')
which_model = cmd_parser.parse_args().model

name_of_run = 'schism_test_v02'
description_of_run = ''

# input description
path_to_hindcast = 'C:\\Users\\laurins\\Documents\\data\\input'
file_mask = 'schism_marl20170101_00z_3D.nc'

schism_crs = proj.CRS.from_proj4('+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
wgs84_crs = proj.CRS.from_epsg(4326)
transformer = proj.Transformer.from_crs(wgs84_crs, schism_crs, always_xy=True)

# output description
path_to_output =  'C:\\Users\\laurins\\Documents\\data\\output'
output_step_size = None

# model description (solver, release, etc.name_of_run)

release_points = np.array([
       [173.786093  , -41.16104113],
       [173.8069543 , -41.15268389],
       [173.8240536 , -41.15088994],
       [173.8394022 , -41.14940125],
       [173.8627361 , -41.13468759],
       [173.8588299 , -41.11545518],
       [173.8414849 , -41.16840586],
       [173.8252307 , -40.99551537],
       [173.7977117 , -41.0284134 ],
       [173.8563229 , -41.05863639],
       [173.9248689 , -41.10665806],
       [173.962755  , -41.13731895],
       [173.9628523 , -41.11518312],
       [174.0404062 , -41.05026245],
       [174.0053709 , -41.03108366],
       [173.9741537 , -41.01266866],
       [173.9882097 , -40.98866365],
       [173.9200421 , -40.97321731],
       [173.963519  , -40.94149451],
       [173.9313326 , -40.92225832],
       [173.9810676 , -40.89785012],
       [173.9006751 , -40.86638754],
       [174.0972068 , -41.02928592],
       [174.2013162 , -41.01473834],
       [174.1787848 , -41.04393466],
       [174.0854508 , -41.17965962],
       [173.9946204 , -41.18222324],
       [173.8869924 , -41.17131367],
       [173.8869924 , -41.17131367],
       [173.8821175 , -41.1505058 ],
       [173.8821175 , -41.1505058 ],
       [173.8704572 , -41.17645025],
       [173.8962498 , -41.11146032],
       [173.8638549 , -41.21179852],
       [173.8638549 , -41.21179852],
       [173.9329508 , -41.07649106],
       [173.8833396 , -41.07830376],
       [173.8833396 , -41.07830376]
       ]) # lon, lat

release_points_schism = transformer.transform(release_points[:,0], release_points[:,1])
pulse_size = np.logspace(3,7,9,dtype=int)

max_model_duration = 1 # days
# care:
# -----
# ot represents the time step as sub steps. hence only ints allowed.
# data time_step is 1800s (30min)
model_time_step = 60 # seconds (1 min)

RK_order = 4

critical_resuspension_vel = 0


if which_model=='oceantracker':
    #%%
    # Oceantracker
    # ------------

    from oceantracker import main
    from oceantracker.post_processing.read_output_files import load_output_files 
    from oceantracker.util import json_util
    # from oceantracker.post_processing.plotting import plot_tracks
    #%%

    for pulse in pulse_size:
        params = {
            "shared_params": {
                "output_file_base": name_of_run + '_ot',
                "root_output_dir": path_to_output
            },
            "reader": {
                "class_name": "oceantracker.reader.schism_reader.SCHSIMreaderNCDF",
                "input_dir": path_to_hindcast,
                "file_mask": file_mask,
            },
            "base_case_params": {
                "run_params": {
                    "user_note": description_of_run,
                    "duration": max_model_duration*24*60*60,
                    "write_tracks": False
                },
                "dispersion": {
                    "A_H": 0.1
                },
                # "tracks_writer": {
                #     "class_name": "oceantracker.tracks_writer.track_writer_compact.FlatTrackWriter",
                # },
                "solver": {
                    # "model_timestep": 1800.0,
                    "n_sub_steps": int(1800/model_time_step),
                    "RK_order": RK_order,
                    "screen_output_step_count": 10,
                },
                "particle_release_groups": [
                    {
                        "points": list([list(item) for item in np.array(release_points_schism).swapaxes(0,1)]),
                        "pulse_size": int(pulse/len(release_points)),
                        "release_interval": 0
                    }
                ],
                "particle_properties": [],
                "trajectory_modifiers": [
                    {
                        "class_name": "oceantracker.trajectory_modifiers.resuspension.BasicResuspension",
                        "critical_friction_velocity": critical_resuspension_vel
                    }
                ],
                "fields": [
                {
                    "class_name": "oceantracker.fields.friction_velocity.FrictionVelocity"
                }
            ],
            }
        }

        runInfo = main.run(params)

        #%%
        caseInfoFile = load_output_files.get_case_info_file_from_run_file(runInfo[0])
        caseInfo = json_util.read_JSON(caseInfoFile)
        total_time = caseInfo['run_info']['model_run_duration']
        total_time = total_time.split(':')
        total_time = int(total_time[0])*3600 + int(total_time[1])*60 + float(total_time[2])

        
        # print(f"| Number of particles | Total model time [m] |")
        # print(f"| {pulse: 19} | {total_time/60: 20.1f} |")

        with open(os.path.join(path_to_output,name_of_run+'.txt'), 'a') as f:
            f.write(f"{which_model},{pulse},{total_time}\n")

        # track_data = load_output_files.load_particle_track_vars(caseInfoFile, var_list=['tide', 'water_depth'])

        # plot_tracks.plot_tracks(track_data, show_grid=True,plot_file_name=os.path.join(path_to_output,name_of_run+'ot.png'))
        # plot_tracks.animate_particles(track_data,
        #     show_grid=True,show_dry_cells=True)

#%%
# OceanDrift
# ----------

if which_model == 'opendrift':
    #%%
    from opendrift.readers import reader_schism_native
    from opendrift.readers import reader_global_landmask
    # from opendrift.models.oceandrift import OceanDrift
    from opendrift.models.sedimentdrift import SedimentDrift
    from datetime import timedelta

    for pulse in pulse_size:
    #%%
        o = SedimentDrift(loglevel=100)  # Set loglevel to 0 for debug information

        reader_landmask = reader_global_landmask.Reader()
        # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
        proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
        schism_native = reader_schism_native.Reader(
                filename = os.path.join(path_to_hindcast,file_mask),
            proj4 = proj4str_nztm,
            use_3d = True)

        o.add_reader([reader_landmask,schism_native])

        # prevent opendrift from making a new dynamical landmask with global_landmask
        o.set_config('general:use_auto_landmask', False)
        if RK_order == 4:
            o.set_config('drift:advection_scheme', 'runge-kutta4')
        elif RK_order == 1:
            o.set_config('drift:advection_scheme', 'euler')
        else:
            print('Error: RK_order must be 1 or 4')
            break   
        o.set_config('drift:horizontal_diffusivity',0.1) 
        o.set_config('vertical_mixing:resuspension_threshold', critical_resuspension_vel)
        o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')


        # Seed elements at defined positions, depth and time
        for point in release_points:
            o.seed_elements(lon=point[0], lat=point[1], radius=0,
                        number=int(pulse/len(release_points)),
                        z=0,
                        time=schism_native.start_time)



        #%%
        #%timeit
        o.run(end_time=schism_native.start_time + timedelta(days=max_model_duration), 
            time_step=900)
            #outfile=os.path.join(path_to_output,name_of_run+'od.nc'))
        
        with open(os.path.join(path_to_output,name_of_run+'.txt'), 'a') as f:
            f.write(f"{which_model},{pulse},{o.timing['total time'].total_seconds()}\n")

        # print(f"| Number of particles | Total model time [m] |")
        # print(f"| {pulse: 19} | {o.timing['total time'].total_seconds()/60: 20.1f} |")

        # o.plot(fast=True,filename=os.path.join(path_to_output,name_of_run+'od.png'))
        # o.animation(fast=True,filename=os.path.join(path_to_output,name_of_run+'od.mp4'))
