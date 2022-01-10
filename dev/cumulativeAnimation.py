import rasterio
import dask.array as da
from dask_rasterio import read_raster, write_raster
import os
import fnmatch
import numpy as np
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthpy.spatial as es
import seaborn as sns
import matplotlib.pyplot as plt
import earthpy.plot as ep
import imageio

# Directory where storms are located.
input_dir = r'Z:\LWI_StageIV\Hurricane Katrina 2005'
movie_dir = 'Z:\LWI_StageIV\!Accumulated'

# Animate Function inputs
# output_dir= r'Z:\LWI_StageIV\test\projected\iterate'
# crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
# dst_crs = 'EPSG:4326'
# img_dir = os.path.join(output_dir, 'img')
# stormName = projected_dir.split('\\')[-2]
# movieFilename = os.path.join(movie_dir, f'{stormName}-accum.gif')

def crop_raster(src, crop_shp):
    cropped_raster, cropped_meta = es.crop_image(src, crop_shp)
    array = cropped_raster
    array[array==0] = np.nan    
    array[array==9999] = np.nan
    src.close()
    return array

def create_image(array, title, output_dir, merge_dir, hours):
    fig, ax = plt.subplots(figsize = (20, 10))
    im = ax.imshow(array, cmap='rainbow', vmin=.1, vmax=400)
    ep.colorbar(im)
    # title = merge_dir[i].split('\\')[-1]
    ax.set(title=f"{merge_dir[0]} to {title} Cumulative Precip (mm) {hours} Hours")
    ax.set_axis_off()
    img_filename = output_dir + f'\\{title}-accum.png'
    plt.savefig((img_filename))
    plt.close()

def AnimateCumulative(input_dir, movie_dir):

    # Animate Function inputs
    projected_dir = os.path.join(input_dir, 'projected' )
    output_dir = os.path.join(projected_dir, 'iterate' )
    # crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
    # dst_crs = 'EPSG:4326'
    img_dir = os.path.join(output_dir, 'img')
    stormName = input_dir.split('\\')[-1]
    movieFilename = os.path.join(movie_dir, f'{stormName}-accum.gif')
    #  Get all files with extension .01h (the default pattern for uncompressed 1hr precip data from UCAR EOL).
    # Script is not accounting for potential naming patterns for AK and PR.
    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(projected_dir),'*.01h'): 
        merge_dir.append((os.path.join(projected_dir, filename)))

    merge_dir.sort()

    with rasterio.open(merge_dir[0]) as src:
            profile=src.profile
    hours = 0
    map2array=[]
    for raster in merge_dir:
        hours += 1
        outputFilename = raster.split('\\')[-1]
        # print(f'adding {raster} to dask array list: map2array')
        src = rasterio.open(raster)
        array = src.read(1)
        # array = crop_raster(src, crop_shp)
        array[array==0] = np.nan    
        array[array==9999] = np.nan
        map2array.append(da.from_array(array, chunks=array.shape))
        if len(map2array) == 1:
            # First array element, dont accumulate.
            # Output Image
            sns.set_style("white")
            sns.set(font_scale=1.5)
            # Check whether the image and projection paths exists.
            # Create a new directory if not isExists.
            isExist_out = os.path.exists(output_dir)
            if not isExist_out:
                os.makedirs(output_dir)
            isExist_img = os.path.exists(img_dir)
            if not isExist_img:
                os.makedirs(img_dir)
            # Crop projected_raster
            # array = crop_raster(src, crop_shp)
            # Create and save Image
            create_image(array, outputFilename, img_dir, merge_dir, hours)
        else: 
            # Accumulate and save Iterative Raster.
            ds_stack = da.stack(map2array)
            print (f'writing Iterative Accumulation Raster to {output_dir}\\{outputFilename}.tif')
            write_raster(f'{output_dir}\\{outputFilename}-accum.tif', da.nansum(ds_stack,0), **profile)
            # Crop Raster
            src_accum = rasterio.open(f'{output_dir}\\{outputFilename}-accum.tif')
            # array_accum = crop_raster(src_accum, crop_shp)
            array_accum = src_accum.read(1)
            array_accum[array_accum==0] = np.nan    
            array_accum[array_accum==9999] = np.nan
            # Create and Save Image
            create_image(array_accum, outputFilename, img_dir, merge_dir, hours)
            # Empty Array
            map2array = []
            # Add the new iterative accumulation raster to first element of the array.
            map2array.append(da.from_array(array_accum, chunks=array_accum.shape))
            # The next iteration of the 'for raster in merge_dir:' loop will add the second element to make the next iterative accumulation.

    # Create Movie
    print (f'Creating gif movie: {movieFilename}')
    imgList=[]
    for img_file in fnmatch.filter(os.listdir(img_dir),'*.png'): 
        imgList.append((os.path.join(img_dir, img_file)))
    imgList.sort()

    imagesArray = []
    for imageFile in imgList:
        # print(f'adding {imageFile} to imagesArray.')
        imagesArray.append(imageio.imread(imageFile))
    imageio.mimsave(movieFilename, imagesArray, duration=0.1)
    print ('<----- Done. ----->\n')

AnimateCumulative(input_dir, movie_dir)