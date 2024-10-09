from parcels import (
    AdvectionRK4,
    AdvectionRK4_3D,
    FieldSet,
    JITParticle,
    ParticleSet,
)


input_datasets = {
        # 'input_base_dir': os.path.abspath('/hpcfreenas/hindcast'),
        'input_base_dir': os.path.abspath('/scratch/local1'),
    }

input_datasets['rom'] = {
    'path_to_hindcast': os.path.join(
        input_datasets['input_base_dir'],
        'ROMS/doppio_bay_02' if sys.platform == 'linux' else 'rom'
        ),
    'file_mask': 'doppio_his_201*.nc',
    # 'transformer': proj.Transformer.from_crs(
    #     proj.CRS.from_epsg(4326), # WGS84
    #     proj.CRS.from_epsg(32619), # UTM19N
    #     always_xy=True),
    'data_dt': 3600, # seconds
    'release_points_lon_lat': np.array([
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
        [-70.09004307,  38.54977471],
        [-69.9444851 ,  38.71378714],
        [-69.84821837,  42.92174757],
        [-69.80693744,  40.67720343],
        [-68.10444023,  41.69506538],
        [-67.62495283,  41.95124591],
        [-64.79557692,  42.78925352],
        [-64.51109065,  43.15402752],
        [-64.26535265,  39.81402787],
        [-63.43641926,  41.95324376],
        [-63.04835776,  43.70211662],
        [-62.37610229,  41.41986588],
        [-61.84710945,  43.14830997],
        [-61.12644699,  42.29980774],
        ])
}