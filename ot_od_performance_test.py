#%%
import argparse
import pyproj as proj
import numpy as np
import os
import pickle
import sys

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
cmd_parser.add_argument('--dataset', type=str, default='rom', help='Which dataset to run. Options are "schism_small","schism_large" and "rom".')
cmd_parser.add_argument('--output', type=int, default=0 , help='Time step size for the output.')

which_model = cmd_parser.parse_args().model
which_dataset = cmd_parser.parse_args().dataset
output_step_size = cmd_parser.parse_args().output

name_of_run = 'full_dataset_test_v04'

# input dictionary
if sys.platform == 'win32':
    input_datasets = {
        'input_base_dir': os.path.abspath('C:\\Users\\laurins\\Documents\\data\\input'),
    }
elif sys.platform == 'linux':
    input_datasets = {
        'input_base_dir': os.path.abspath('/hpcfreenas/hindcast'),
    }

# schism small aka coarse NZ
input_datasets['schism_small'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'OceanNumNZ-2022-06-20/final_version/2017/01' if sys.platform == 'linux' else 'schism_small'
        ),
    'file_mask': 'NZfinite201701*.nc',
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_proj4('+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'), # NZTM
        always_xy=True),
    'data_dt': 3600, # seconds
    'release_points_lon_lat': np.array([
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
}

# schism large aka marlbourough sounds
input_datasets['schism_large'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'MarlbroughSounds_hindcast_10years_BenPhd_2019ver/2017' if sys.platform == 'linux' else 'schism_large'
        ),
    'file_mask': 'schism_marl201701*.nc',
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_proj4('+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'), # NZTM
        always_xy=True),
    'data_dt': 1800, # seconds
    'release_points_lon_lat': np.array([
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

}

# rom
input_datasets['rom'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'ROMS/doppio_bay_02' if sys.platform == 'linux' else 'rom'
        ),
    'file_mask': 'doppio_his_201*.nc',
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_epsg(32619), # UTM19N
        always_xy=True),
    'data_dt': 3600, # seconds
    'release_points_lon_lat': np.array([
        # [-78.06824664,  38.41109756], # outside domain
        # [-77.48956078,  35.87834191], # outside domain
        # [-76.9632904 ,  39.24838291], # outside domain
        # [-76.45835255,  35.68510219], # outside domain
        [-75.51165512,  38.31082864],
        [-74.8860462 ,  35.44369297],
        [-74.52862058,  34.2950443 ],
        [-74.21067673,  34.67342632],
        [-73.95323861,  35.41533207],
        [-73.46990185,  35.73379049],
        [-73.18779878,  39.79203454],
        [-72.84653146,  38.95630874],
        [-72.63457538,  36.91732252],
        [-72.59191209,  40.63632652],
        [-72.19871125,  35.34782645],
        [-71.16154411,  36.63844384],
        # [-71.15314662,  42.11431184],
        # [-70.46635119,  43.52036114],
        [-70.09004307,  38.54977471],
        [-69.9444851 ,  38.71378714],
        [-69.84821837,  42.92174757],
        [-69.80693744,  40.67720343],
        [-68.10444023,  41.69506538],
        [-67.62495283,  41.95124591],
        [-67.08252378,  38.31765205],
        [-66.85770073,  42.88834616],
        [-65.48411736,  42.97607939],
        [-65.31633613,  45.24194   ],
        [-64.79557692,  42.78925352],
        [-64.51109065,  43.15402752],
        [-64.26535265,  39.81402787],
        # [-63.5384388 ,  44.58749064],
        [-63.43641926,  41.95324376],
        [-63.04835776,  43.70211662],
        [-62.37610229,  41.41986588],
        [-61.84710945,  43.14830997],
        [-61.12644699,  42.29980774],
        [-60.99873369,  42.50516744]
        ])

}

# output description
path_to_output =  os.path.join(
    'C:\\Users\\laurins\\Documents\\data\\output' if sys.platform == 'win32' else '/home/laurins/data/output',
    name_of_run)
os.makedirs(path_to_output, exist_ok=True)
# output_step_size = 0 # in sec, 0 means no output

# model description (solver, release, etc.name_of_run)

pulse_size = np.logspace(3,6,4,dtype=int)

max_model_duration = 10 # days
# care
# -----
# ot represents the time step as sub steps. hence only ints allowed.
model_time_step = 60*5 # seconds (5 min)

RK_order = 4

critical_resuspension_vel = 0

for pulse in pulse_size:
    output_file_base = f'{name_of_run}_data_{which_dataset}_particle_{pulse}_output_{output_step_size}'

    if which_model=='oceantracker':
        # Oceantracker
        # ------------

        from oceantracker import main
        from oceantracker.post_processing.read_output_files import load_output_files 
        from oceantracker.util import json_util
        from oceantracker.post_processing.plotting import plot_tracks

        params = {
            "shared_params": {
                "output_file_base": f'{output_file_base}_ot',
                "root_output_dir": path_to_output
            },
            "reader": {
                "class_name": "oceantracker.reader.schism_reader.SCHSIMreaderNCDF" if ('schism' in which_dataset) else "oceantracker.reader.dev_ROMS_reader.ROMsNativeReader",
                "input_dir": input_datasets[which_dataset]['path_to_hindcast'],
                "file_mask": input_datasets[which_dataset]['file_mask'],
            },
            "base_case_params": {
                "run_params": {
                    "duration": max_model_duration*24*60*60,
                    "write_tracks": False if output_step_size==0 else True,
                },
                "dispersion": {
                    "A_H": 0.1,
                    "A_V": 0.01
                },
                "tracks_writer": {
                    "class_name": "oceantracker.tracks_writer.track_writer_compact.FlatTrackWriter",
                    "output_step_count": int(output_step_size/model_time_step) if output_step_size!=0 else 1
                },
                "solver": {
                    "n_sub_steps": int(input_datasets[which_dataset]['data_dt']/model_time_step),
                    "RK_order": RK_order,
                    "screen_output_step_count": int(output_step_size/model_time_step) if output_step_size!=0 else 1
                },
                "particle_release_groups": [
                    {
                        "points": list([list(item) for item in np.array(
                            input_datasets[which_dataset]['transformer'].transform(
                                input_datasets[which_dataset]['release_points_lon_lat'][:,0],
                                input_datasets[which_dataset]['release_points_lon_lat'][:,1])
                            ).swapaxes(0,1)]),
                        "pulse_size": int(pulse/len(input_datasets[which_dataset]['release_points_lon_lat'])),
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

        caseInfoFile = load_output_files.get_case_info_file_from_run_file(runInfo[0])
        caseInfo = json_util.read_JSON(caseInfoFile)
        total_time = caseInfo['run_info']['model_run_duration']
        total_time = total_time.split(':')
        total_time = int(total_time[0])*3600 + int(total_time[1])*60 + float(total_time[2])
        
        if output_step_size != 0:
            track_data = load_output_files.load_particle_track_vars(caseInfoFile, var_list=['tide', 'water_depth'])
            plot_tracks.plot_tracks(track_data, show_grid=True,
                plot_file_name=os.path.join(
                    path_to_output,
                    f'{output_file_base}_ot',
                    'tracks.png'
                )
            )
            # plot_tracks.animate_particles(track_data,
            #     show_grid=True,show_dry_cells=True)

#%%
# OceanDrift
# ----------

    if which_model == 'opendrift':
        from opendrift.readers import reader_schism_native
        from opendrift.readers import reader_ROMS_native
        from opendrift.readers import reader_global_landmask
        # from opendrift.models.oceandrift import OceanDrift
        from opendrift.models.sedimentdrift import SedimentDrift
        from datetime import timedelta

        o = SedimentDrift(loglevel=100)  # Set loglevel to 0 for debug information


        if 'schism' in which_dataset:
            reader_landmask = reader_global_landmask.Reader()
            # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
            proj4str_nztm = '+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
            schism_native = reader_schism_native.Reader(
                filename = os.path.join(
                    input_datasets[which_dataset]['path_to_hindcast'],
                    input_datasets[which_dataset]['file_mask']
                    ),
                proj4 = proj4str_nztm,
                use_3d = True)
            o.add_reader([reader_landmask,schism_native])
            o.set_config('general:use_auto_landmask', False)
            dataset_start_time = schism_native.start_time
        else: 
            roms_reader = reader_ROMS_native.Reader(
                filename = os.path.join(
                    input_datasets[which_dataset]['path_to_hindcast'],
                    input_datasets[which_dataset]['file_mask']
                    )
            )
            o.add_reader(roms_reader)
            dataset_start_time = roms_reader.start_time

        # prevent opendrift from making a new dynamical landmask with global_landmask
        
        if RK_order == 4:
            o.set_config('drift:advection_scheme', 'runge-kutta4')
        elif RK_order == 1:
            o.set_config('drift:advection_scheme', 'euler')
        else:
            print('Error: RK_order must be 1 or 4')
            break   
        
        # stokes drift
        o.set_config('drift:stokes_drift', False)

        # horizontal mixing
        o.set_config('drift:horizontal_diffusivity',0.1) 

        # vertical mixing      
        # the constant one (which would be the fairest) would require some hacking
        # to define an approriate environment in all cases to get it to work
        # hence we use the default non-constant one. 
        # the comp cost are not expected to be significant
        # o.set_config('vertical_mixing:diffusivitymodel', 'constant')
        # o.set_config('environment:fallback:ocean_vertical_diffusivity', 0.01)
        o.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')

        # resuspension
        o.set_config('vertical_mixing:resuspension_threshold', critical_resuspension_vel)



        # Seed elements at defined positions, depth and time
        for point in input_datasets[which_dataset]['release_points_lon_lat']:
            o.seed_elements(lon=point[0], lat=point[1], radius=0,
                number=int(pulse/len(input_datasets[which_dataset]['release_points_lon_lat'])),
                z=0,time=dataset_start_time)


        os.makedirs(os.path.join(
            path_to_output,f'{output_file_base}_od'),
            exist_ok=True
            )

        o.run(end_time=dataset_start_time + timedelta(days=max_model_duration), 
            time_step=model_time_step,
            time_step_output=output_step_size if output_step_size != 0 else max_model_duration*24*60*60,
            outfile=os.path.join(
                path_to_output,f'{output_file_base}_od','tracks.nc') if output_step_size != 0 else None,
            )

        with open(os.path.join(path_to_output,f'{output_file_base}_od','timing.pkl'), 'wb') as f:
            pickle.dump(o.timing, f)

        total_time = o.timing['total time'].total_seconds()

        if output_step_size != 0:
            o.plot(fast=True,filename=os.path.join(
                path_to_output,f'{output_file_base}_od','tracks.png'
                )
            )
            # o.animation(fast=True,filename=os.path.join(path_to_output,name_of_run+'_'+which_dataset+'_od.mp4'))

    with open(os.path.join(path_to_output,name_of_run+'.txt'), 'a') as f:
        f.write(f"{which_model},{which_dataset},{pulse},{output_step_size},{total_time}\n")

