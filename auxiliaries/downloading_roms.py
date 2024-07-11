# %%
import os
import threddsclient
import urllib.request
import lxml

# %%
querry_url = 'https://tds.marine.rutgers.edu/thredds/catalog/roms/doppio/2017_da/his/files/catalog.html'
source_url = 'https://tds.marine.rutgers.edu/thredds/fileServer/roms/doppio/2017_da/his/files/'

# source_url = 'https://tds.marine.rutgers.edu/thredds/catalog/roms/doppio/2017_da/his/catalog.html'
# destination_dir = r'C:\Users\laurins\Documents\data\input\rom'
destination_dir = r'/scratch/local1/ROMS/doppio_bay_03'

files = []
for ds in threddsclient.crawl(querry_url, depth=1):
    files.append(ds.name)

# traverse_thredds(source_url, destination_dir, 1, True, True)

# %%
# make sure that destination dir exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# %%
to_download = [file for file in files if ('0000_0001' in file)*('201801' in file)]
to_download

# %%
to_download = [file for file in files if ('0000_0001' in file)*('201801' in file)]
for file_name in to_download:
    print(file_name)
    urllib.request.urlretrieve(source_url+file_name, os.path.join(destination_dir,file_name))


