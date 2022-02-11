from multiprocessing.connection import wait
import rasterio
import os, fnmatch
import time

# stormDir = r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\NonTropicalStorms"
stormDir = r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\fixCorruptNTSRain"
storm_dirs = next( os.walk(stormDir) )[1][1:]
badList=[]
for storm in storm_dirs:
    print (f'\nStorm: {storm}')
    input_dir = os.path.join(stormDir, storm)
    merge_dir=[]
    
    for filename in fnmatch.filter(os.listdir(input_dir),'*.01h'): 
        merge_dir.append((os.path.join(input_dir, filename)))
    for filename in fnmatch.filter(os.listdir(input_dir),'*.grb2'): 
        merge_dir.append((os.path.join(input_dir, filename)))

    merge_dir.sort()
    
    for raster in merge_dir:
        try:
            with rasterio.open(raster, 'r') as src:
                src.close()
        except:
            time.sleep(2)
            try:
                with rasterio.open(raster, 'r') as src:
                    src.close()
            except:
                print(f'bad bad {src}')
                badList.append(src)
print (badList)