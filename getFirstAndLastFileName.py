import rasterio
import os, fnmatch
import time

# stormDir = r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\NonTropicalStorms"
stormDir = r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\!Corrupt_NTS"
storm_dirs = next( os.walk(stormDir) )[1][1:]
badList=[]
for storm in storm_dirs:
    # print (f'\nStorm: {storm}')
    input_dir = os.path.join(stormDir, storm)
    merge_dir=[]
    
    for filename in fnmatch.filter(os.listdir(input_dir),'*.01h'): 
        merge_dir.append(filename)
    for filename in fnmatch.filter(os.listdir(input_dir),'*.grb2'): 
        merge_dir.append(filename)

    merge_dir.sort()
    print (storm, merge_dir[0].split(".")[1][:-2], merge_dir[-1].split(".")[1][:-2] )