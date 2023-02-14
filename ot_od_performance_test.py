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
cmd_parser.add_argument('--model', type=str, default='opendrift', help='Which model to run. Options are "opendrift" and "oceantracker".')
which_model = cmd_parser.parse_args().model

name_of_run = 'schism_test'
description_of_run = ''

# input description
path_to_hindcast = '/home/ls/data/estuar/cawthron/'
file_mask = 'schism_marl20170101_00z_3D.nc'

schism_crs = proj.CRS.from_proj4('+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
wgs84_crs = proj.CRS.from_epsg(4326)
transformer = proj.Transformer.from_crs(wgs84_crs, schism_crs, always_xy=True)

# output description
path_to_output =  '/home/ls/data/estuar/cawthron/output'
output_step_size = None

# model description (solver, release, etc.name_of_run)

release_points = [174.046669,-40.928116] # lon, lat
release_points_schism = transformer.transform(release_points[0], release_points[1])
pulse_size = 1000 

max_model_duration = 1 # days
# care:
# -----
# ot represents the time step as sub steps. hence only ints allowed.
# data time_step is 1800s (30min)
model_time_step = 900 # seconds (15 min)

critical_resuspension_vel = 0

# TO DO:






if which_model=='oceantracker':
    #%%
    # Oceantracker
    # ------------

    from oceantracker import main
    from oceantracker.post_processing.read_output_files import load_output_files 
    from oceantracker.post_processing.plotting import plot_tracks
    #%%
    params = {
        "shared_params": {
            "output_file_base": name_of_run,
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
            },
            "dispersion": {
                "A_H": 0.1
            },
            "tracks_writer": {
                "class_name": "oceantracker.tracks_writer.track_writer_compact.FlatTrackWriter",
            },
            "solver": {
                # "model_timestep": 1800.0,
                "n_sub_steps": int(1800/model_time_step),
                "RK_order": 1 # open drift only supports 1st order
            },
            "particle_release_groups": [
                {
                    "points": [
                        [
                            release_points_schism[0],
                            release_points_schism[1]
                        ]
                    ],
                    "pulse_size": pulse_size,
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

    #%%
    #%timeit
    # Runningx``
    runInfo = main.run(params)

    #%%
    caseInfoFile = load_output_files.get_case_info_file_from_run_file(runInfo[0])

    track_data = load_output_files.load_particle_track_vars(caseInfoFile, var_list=['tide', 'water_depth'])

    plot_tracks.plot_tracks(track_data, show_grid=True,plot_file_name=os.path.join(path_to_output,name_of_run+'ot.png'))
    plot_tracks.animate_particles(track_data,
        show_grid=True,show_dry_cells=True)

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

    #%%
    # o = OceanDrift(loglevel=100)  # Set loglevel to 0 for debug information
    o = SedimentDrift(loglevel=100)  # Set loglevel to 0 for debug information

    reader_landmask = reader_global_landmask.Reader()
    # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
    proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    schism_native = reader_schism_native.Reader(
        filename = path_to_hindcast + file_mask,
        proj4 = proj4str_nztm,
        use_3d = True)

    o.add_reader([reader_landmask,schism_native])

    # prevent opendrift from making a new dynamical landmask with global_landmask
    o.set_config('general:use_auto_landmask', False)

    o.set_config('drift:horizontal_diffusivity',0.1) 
    o.set_config('vertical_mixing:resuspension_threshold', critical_resuspension_vel)
    o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')


    # Seed elements at defined positions, depth and time
    o.seed_elements(lon=release_points[0], lat=release_points[1], radius=0, number=pulse_size,
                    z=0,
                    time=schism_native.start_time)



    #%%
    #%timeit
    o.run(end_time=schism_native.start_time + timedelta(days=max_model_duration), 
          time_step=900, 
          outfile=os.path.join(path_to_output,name_of_run+'.nc'))

    print(o)
    o.plot(fast=True,filename=os.path.join(path_to_output,name_of_run+'od.png'))
    o.animation(fast=True,filename=os.path.join(path_to_output,name_of_run+'od.mp4'))
