import argparse
import pyproj as proj
import numpy as np
import os
import pickle
import sys
import time

'''
Here we compare the speed of OpenDrift with the speed of oceantracker03.
We test it on a SCHISM (unstruct) hindcast in Marlborough Sounds.
'''


name_of_run = 'OT_dev050_with_SVML_enabled'

## version 07 and 8 using commit
# commit e2f2929c01d1f627483658a868b3eca7e4c14b17 (HEAD -> dev041, origin/dev041)
# Author: Ross Vennell <ross.vennell@cawthron.org.nz>
# Date:   Mon Feb 5 11:54:03 2024 +1300

#     tidied up setting numba config via env variables, and added numba conf to case info

## version 06 using commit 
# commit b3069288b460d56810af0e50652643e8f792470b (HEAD -> dev041)
# Author: Ross Vennell <ross.vennell@cawthron.org.nz>
# Date:   Thu Dec 21 11:06:36 2023 +1300
# 
#     auto detecting gerenic netcdf format

# Preparation (def. params, model-set, etc)
# =========================================
# Parameterisation (for both models)
# ----------------------------------

cmd_parser = argparse.ArgumentParser(description='Run a test of the performance of OpenDrift and OceanTracker.')
cmd_parser.add_argument('--model', type=str, default='parcels', help='Which model to run. Options are "opendrift" and "oceantracker".')
cmd_parser.add_argument('--dataset', type=str, default='nemo', help='Which dataset to run. Options are "schism_small","schism_large" and "rom".')
cmd_parser.add_argument('--output', type=int, default=0 , help='Time step size for the output.')

which_model = cmd_parser.parse_args().model
which_dataset = cmd_parser.parse_args().dataset
output_step_size = cmd_parser.parse_args().output

# input dictionary
if sys.platform == 'win32':
    input_datasets = {
        'input_base_dir': os.path.abspath('C:\\Users\\laurins\\Documents\\data\\input'),
    }
elif sys.platform == 'linux':
    input_datasets = {
        # 'input_base_dir': os.path.abspath('/hpcfreenas/hindcast'),
        'input_base_dir': os.path.abspath('/scratch/local1'),
    }

# schism small aka coarse NZ
input_datasets['schism_estuary'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'hzg' if sys.platform == 'linux' else 'schism_small'
        ),
    'file_mask': 'schout*.nc',
    'oceantracker_reader': "SCHISMreaderNCDF",
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_epsg(25832), # UTM32N
        always_xy=True),
    'data_dt': 3600, # seconds
    'release_points_lon_lat': np.array([
       [ 9.03187469, 53.86750645],
       [ 9.55826941, 53.61036541],
       [ 8.21109806, 53.99972419],
       [ 8.54838929, 53.97008742],
       [ 9.91824767, 53.54178875],
       [ 8.77125669, 53.98610574],
       [ 9.91826257, 53.54177965],
       [ 9.03187469, 53.86750645],
       [ 9.55826941, 53.61036541],
       [ 8.21109806, 53.99972419],
    ]),
    'release_points_xy': np.array([
        [502096, 5968781],
        [536935, 5940317],
        [448288, 5983779],
        #
        [470376, 5980288],
        [560849, 5932935],
        #
        [485001, 5982000],
        [560850, 5932934],
        #
        [470376, 5980287],
        [560849, 5932934],
        #
        [485000, 5981999],
        ])
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
    'oceantracker_reader': "ROMS_reader",
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_epsg(32619), # UTM19N
        always_xy=True),
    'data_dt': 3600, # seconds
    'release_points_lon_lat': np.array([
            [-74.8860462 ,  35.44369297],
            [-74.21067673,  34.67342632],
            [-72.84653146,  38.95630874],
            [-72.59191209,  40.63632652],
            [-71.16154411,  36.63844384],
            [-69.9444851 ,  38.71378714],
            [-69.80693744,  40.67720343],
            [-64.51109065,  43.15402752],
            [-63.43641926,  41.95324376],
            [-62.37610229,  41.41986588],
            ])
}

# nemo
input_datasets['nemo'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'NEMO/baltic'
        ),
    'file_mask': '*.nc',
    'oceantracker_reader': "GLORYSreader",
    'transformer': proj.Transformer.from_crs(
        proj.CRS.from_epsg(4326), # WGS84
        proj.CRS.from_epsg(25833), # UTM33N
        always_xy=True),
    'data_dt': 3600, # ,
    'release_points_lon_lat': np.array([
       [26.24914652, 53.62084284],
       [11.74138304, 57.59627221],
       [ 9.88378601, 59.12949598],
       [30.07541881, 65.19897143],
       [29.32377253, 62.96581345],
       [26.87092123, 58.95689007],
       [12.9142767, 60.57033272 ],
       [20.37102819, 64.97765214],
       [16.58614791, 54.51054369],
       [24.33259039, 55.17697955],
    #    [28.98996948, 64.28547982],
    #    [10.10919548, 61.32635904],
    #    [24.19031533, 54.1708471 ],
    #    [26.40140448, 61.32829611],
    #    [25.54562022, 53.65127282],
    #    [14.40421594, 65.31829057],
    #    [ 9.2518973, 59.43119579 ],
    #    [12.30778475, 65.0992816 ],
    #    [20.31473819, 65.07813115],
    #    [22.50382744, 60.61403735],
    #    [10.3043585, 54.0901189  ],
    #    [ 9.51512965, 59.50471693],
    #    [26.93017637, 57.00587859],
    #    [16.01582867, 64.98688646],
    #    [21.9730505, 53.04142688 ],
    #    [13.03814926, 60.22205362],
    #    [21.25043432, 62.89193372],
    #    [12.0119608, 53.26718728 ],
       ])
}
# input_datasets['nemo']['release_points_lat_lon'] = input_datasets['nemo']['release_points_lon_lat'][:, [1, 0]]

# output description
path_to_output =  os.path.join(
    # 'C:\\Users\\laurins\\Documents\\data\\output' if sys.platform == 'win32' else '/home/laurins/data/output',
    'C:\\Users\\laurins\\Documents\\data\\output' if sys.platform == 'win32' else '/scratch/local1/speed_test_output',
    name_of_run)
os.makedirs(path_to_output, exist_ok=True)

# model configuration
pulse_size = np.logspace(1,6,6,dtype=int) 
max_model_duration = 10 # days
model_time_step = 60*5 # seconds (5 min)
critical_resuspension_vel = 0


for pulse in [10_000_000]: #pulse_size:

    output_file_base = f'{name_of_run}_data_{which_dataset}_particle_{pulse}_output_{output_step_size}'

    if which_model=='parcels':
        from datetime import timedelta
        from glob import glob
        # import xarray as xr

        from parcels import (
            AdvectionRK4,
            AdvectionRK4_3D,
            FieldSet,
            JITParticle,
            ScipyParticle,
            ParticleSet,
        )

        start_time = time.time()        

        if which_dataset == "nemo":
       
            uvw_files = sorted(glob(os.path.join(input_datasets['nemo']['path_to_hindcast'],'*PHY*.nc')))
            mesh_mask = glob(os.path.join(input_datasets['nemo']['path_to_hindcast'],'*coordinates.nc'))
            
            filenames = {
                "U": {"lon": uvw_files[0],
                    "lat": uvw_files[0],
                    "depth": uvw_files[0],
                    "data": uvw_files
                    },
                "V": {"lon": uvw_files[0],
                    "lat": uvw_files[0],
                    "depth": uvw_files[0],
                    "data": uvw_files
                    },
                "W": {"lon": uvw_files[0],
                    "lat": uvw_files[0],
                    "depth": uvw_files[0],
                    "data": uvw_files
                    }
            }
            variables = {
                "U": "uo",
                "V": "vo",
                "W": "wo",
            }

            c_grid_dimensions = {
                "lon": "lon",
                "lat": "lat",
                "depth": "depth",
                "time": "time",
            }

            dimensions = {
                "U": c_grid_dimensions,
                "V": c_grid_dimensions,
                "W": c_grid_dimensions,
            }

            fieldset = FieldSet.from_nemo(filenames, variables, dimensions)


            release_count_multiplier = int(max(pulse/len(input_datasets[which_dataset]['release_points_lon_lat']),1))
            
            pset = ParticleSet(fieldset, JITParticle,
                            lat=np.repeat(input_datasets['nemo']['release_points_lon_lat'][:,1],release_count_multiplier),
                            lon=np.repeat(input_datasets['nemo']['release_points_lon_lat'][:,0],release_count_multiplier),
                            depth=[1]*len(input_datasets['nemo']['release_points_lon_lat'][:,0])*release_count_multiplier
                            )

        elif which_dataset == "rom":
            files = glob(
                os.path.join(
                    input_datasets['rom']['path_to_hindcast'],
                    input_datasets['rom']['file_mask']
                    )
                )
            files = sorted(files)
            files = files[:1]

            timestamps = np.loadtxt(
                'timestamps_of_doppio_first_10d.txt', dtype='datetime64[s]'
                )
            timestamps = np.array([timestamps,])

            start_time = time.time()

            fieldset = FieldSet.from_netcdf(
            filenames = files,
            variables = {
                'U': 'u',
                'V': 'v',
                # 'W': 'w',
                },
            dimensions = {
                'U': {
                    'lon': 'lon_u',
                    'lat': 'lat_u',
                    'depth': 's_rho',
                    'time': 'ocean_time'
                    },
                'V': {
                    'lon': 'lon_v',
                    'lat': 'lat_v',
                    'depth': 's_rho',
                    'time': 'ocean_time'
                    },
                # 'W': {
                #     'lon': 'lon_rho',
                #     'lat': 'lat_rho',
                #     'depth': 's_w',
                #     'time': 'ocean_time'
                #     },
            },
            timestamps = timestamps,
            deferred_load = True,
            # parcels has an issue with overlapping data sets 
            # i.e. if element 0 ends at midnight and element 1 starts
            # at midnight.
            # to avoid rewritting the netCDF I loop as we
            # assume neglicable read/write times anyway
            time_periodic = timedelta(hours=24),
            )

            spawn_points = input_datasets['rom']['release_points_lon_lat'][np.random.choice(np.arange(len(input_datasets['rom']['release_points_lon_lat'])),size=pulse)]
            set = ParticleSet.from_list(
                fieldset=fieldset,  # the fields on which the particles are advected
                pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
                lon=spawn_points[:,0],  # a vector of release longitudes
                lat=spawn_points[:,1],  # a vector of release latitudes
            )

        if output_step_size != 0:
            output_file = pset.ParticleFile(
                name=os.path.join(
                    path_to_output,f'{output_file_base}','tracks.zarr'),  # the file name
                    outputdt=timedelta(seconds=model_time_step*output_step_size),  # the time step of the outputs
            )
            pset.execute(
                AdvectionRK4_3D,  # the kernel (which defines how particles move)
                runtime=timedelta(days=max_model_duration),  # the total length of the run
                dt=timedelta(seconds=model_time_step),  # the timestep of the kernel
                output_file=output_file,
            )

        else:
            pset.execute(
                AdvectionRK4_3D,  # the kernel (which defines how particles move)
                runtime=timedelta(days=max_model_duration),  # the total length of the run
                dt=timedelta(seconds=model_time_step),  # the timestep of the kernel
            )
        
        end_time = time.time()
        total_time = end_time - start_time

    elif which_model=='oceantracker':
        # Oceantracker
        # ------------

        from oceantracker import main
        from read_oceantracker.python.load_output_files import load_track_data
        from oceantracker.util import json_util

        params = {
            "output_file_base": f'{output_file_base}_ot',
            "root_output_dir": path_to_output,
            "screen_output_time_interval": 1e9,
            "max_run_duration": max_model_duration*24*60*60,
            "time_step": model_time_step,
            "write_tracks": False if output_step_size==0 else True,
            "reader": {
                "class_name": input_datasets[which_dataset]['oceantracker_reader'],
                "input_dir": input_datasets[which_dataset]['path_to_hindcast'],
                "file_mask": input_datasets[which_dataset]['file_mask'],
                "time_buffer_size": 6,
            },
            "dispersion": {
                "A_H": 0.1,
                "A_V": 0.01
            },
            "resuspension": {
                "critical_friction_velocity": critical_resuspension_vel
            },
            # "tracks_writer": {
            #     "class_name": "oceantracker.tracks_writer.track_writer_compact.FlatTrackWriter",
            #     "output_step_count": int(output_step_size/model_time_step) if output_step_size!=0 else 1
            # },
            "tracks_writer": {
                "update_interval": model_time_step*output_step_size if output_step_size != 0 else 1
            },
            # "solver": {
            #     "RK_order": RK_order,
            #     "screen_output_step_count": int(output_step_size/model_time_step) if output_step_size!=0 else 1
            # },
            "release_groups": [
                {
                    "name": "default",
                    # "points": list([list(item) for item in np.array(
                    #     input_datasets[which_dataset]['transformer'].transform(
                    #         input_datasets[which_dataset]['release_points_lon_lat'][:,0],
                    #         input_datasets[which_dataset]['release_points_lon_lat'][:,1])
                    #     ).swapaxes(0,1)]),
                    "points": input_datasets[which_dataset]['release_points_lon_lat'],
                    "pulse_size": int(max(pulse/len(input_datasets[which_dataset]['release_points_lon_lat']),1)),
                    "release_interval": 0
                }
            ],
            # "particle_properties": [],
            # "fields": [
            # {
            #     "class_name": "oceantracker.fields.friction_velocity.FrictionVelocity"
            # }
            # ],
        }

        # drop dispersion if the model is nemo
        if which_dataset == 'nemo':
            params['dispersion'] == 0

        
        case_info_path = main.run(params)

        # caseInfoFile = load_output_files.get_case_info_file_from_run_file(runInfo[0])
        case_info = json_util.read_JSON(case_info_path)
        total_time = case_info['run_info']['computation_duration']
        # transform total time from '0:00:26.857793' to seconds
        total_time = total_time.split(':')
        total_time = int(total_time[0])*3600 + int(total_time[1])*60 + float(total_time[2])
        

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

        if 'schism_estuary' in which_dataset:
            reader_landmask = reader_global_landmask.Reader()
            # NZTM proj4 string found at https://spatialreference.org/ref/epsg/nzgd2000-new-zealand-transverse-mercator-2000/
            proj4str_nztm = '+proj=utm +zone=32 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs'
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

        elif 'schism' in which_dataset:
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
        
        # solver rk4
        o.set_config('drift:advection_scheme', 'runge-kutta4')
        
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
                #z=0,
                time=dataset_start_time)


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

        # with open(os.path.join(path_to_output,f'{output_file_base}_od','timing.pkl'), 'wb') as f:
        #     pickle.dump(o.timing, f)

        total_time = o.timing['total time'].total_seconds()

    with open(os.path.join(path_to_output,name_of_run+'.txt'), 'a') as f:
        f.write(f"{which_model},{which_dataset},{pulse},{output_step_size},{total_time}\n")

