import rasterio
import dask.array as da
from dask_rasterio import read_raster, write_raster
import os
import fnmatch
import numpy as np
import geopandas as gpd
import shapely
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthpy.spatial as es
import seaborn as sns
import matplotlib.pyplot as plt
import earthpy.plot as ep
import imageio
# import rasterio.plot

# TODO Implement Cropped Incremental Movie

# fig, ax = plt.subplots(figsize=(15, 15))
# la = gpd.read_file("Z:\GIS\Louisiana.shp")
# bbox = la.total_bounds
# la_extent=bbox[[0,2,1,3]]

# bbox_lwi = shapefile.total_bounds
# lwi_extent = bbox_lwi[[0,2,1,3]]
# fn = r"Z:\LWI_StageIV\Hurricane Katrina 2005\projected\ST4.2005082913.01h"
# title = fn.split('\\')[-1]
# raster = rasterio.open(fn)

# raster_window = raster.window(*bbox_lwi)
# array = raster.read(1, window=raster_window)
# array[array==raster.nodata] = np.nan
# im = plt.imshow(array, extent=lwi_extent, cmap='rainbow')
# cb = plt.colorbar(im, shrink=.6)
# ax.set(title=f"{title} Cumulative Precip (mm)")
# ax.set_axis_off()
# la.boundary.plot(ax=plt.gca(), color='darkgrey')

# Takes a directory of Stage IV precip data from UCAR (uncompressed) and accumulates .01h files to a single geotif.
# An image file of the geotif is also produced to quickly view the result.

# Directory where storms are located.
stormDir = r'Z:\LWI_StageIV'
outputDir= r'Z:\LWI_StageIV\!Accumulated'
# Animate Function inputs
crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
dst_crs = 'EPSG:4326'

def projectRaster(raster_fn, raster_src, projected_dir):
    transform, width, height = calculate_default_transform(
            raster_src.crs, dst_crs, raster_src.width, raster_src.height, *raster_src.bounds)
    kwargs = raster_src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    projected_raster = projected_dir + '\\' + raster_fn.split("\\")[-1]
    with rasterio.open(projected_raster, 'w', **kwargs) as dst:
        for i in range(1, raster_src.count + 1):
            reproject(
                source=rasterio.band(raster_src, i),
                destination=rasterio.band(dst, i),
                src_transform=raster_src.transform,
                src_crs=raster_src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
    return projected_raster, kwargs

def crop_raster(src, crop_shp):
    cropped_raster, cropped_meta = es.crop_image(src, crop_shp)
    array = cropped_raster
    array[array==0] = np.nan    
    array[array==9999] = np.nan
    src.close()
    return array

def cropRasterByMask(src_projected, crop_shp, cropped_raster_fn):
    geom = []
    coord = shapely.geometry.mapping(crop_shp)["features"][0]["geometry"]
    geom.append(coord)

    out_image, out_transform = rasterio.mask.mask(src_projected, geom, crop=True)
    out_meta = src_projected.meta
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(os.path.join(cropped_raster_fn), "w", **out_meta) as dest:
        dest.write(out_image)

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

# Writes out Projected hourly tifs, then writes Cropped hourly Tifs, Then Accumulates to a Single Tif.
def projectCropAccumulate(input_dir, output_dir, crop_shp, dst_crs):
    projected_dir = os.path.join(input_dir, 'projected')
    cropped_dir = os.path.join(input_dir, 'cropped')
    outputFilename = input_dir.split('\\')[-1]+'-cropped.tif'
    # Assumes Directory Name is the name of the storm and wanted name of the output files.
    # I.e: input_dir=r'Z:\LWI_StageIV\Hurricane Cindy 2005' ==> output file will be 'Hurricane Cindy 2005.tif'
    outputFilename = input_dir.split('\\')[-1]+'-cropped.tif'

    # Check if output, projected, and cropped directories exist.
    isExist = os.path.exists(output_dir)
    if not isExist:
            os.makedirs(output_dir)
    isExist = os.path.exists(projected_dir)
    if not isExist:
            os.makedirs(projected_dir)
    isExist = os.path.exists(cropped_dir)
    if not isExist:
            os.makedirs(cropped_dir)
    
    # get all files with extension .01h (the default pattern for uncompressed 1hr precip data from UCAR EOL).
    # Script is not accounting for potential naming patterns for AK and PR.
    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(input_dir),'*.01h'): 
        merge_dir.append((os.path.join(input_dir, filename)))

    merge_dir.sort()

    map2array=[]
    for raster in merge_dir:
        print(f'projecting and cropping {raster}. Then adding to dask array list: map2array')
        src = rasterio.open(raster)
        projected_raster, kwargs = projectRaster(raster, src, projected_dir, dst_crs)
        src_projected = rasterio.open(projected_raster)
        # array, cropped_meta = crop_raster(src, crop_shp)
        # Write Cropped Raster to Disk
        cropped_raster_fn = cropped_dir + '\\' + raster.split("\\")[-1]
        cropRasterByMask(src_projected, crop_shp, cropped_raster_fn)
        cropped_src = rasterio.open(cropped_raster_fn)
        profile = cropped_src.profile
        array = cropped_src.read(1)
        array[array==0] = np.nan    
        array[array==9999] = np.nan
        # with rasterio.open(cropped_raster_fn, 'w', **kwargs) as dest:
        #     dest.write(array.squeeze().astype(rasterio.uint8), 1)
        #     profile = dest.profile
        # map2array.append(read_raster(raster, band=1, block_size=10))
        map2array.append(da.from_array(array, chunks=array.shape))
        src.close()
        src_projected.close()
        cropped_src.close()

    ds_stack = da.stack(map2array)
    print (f'writing Raster to {output_dir}\\{outputFilename}')
    write_raster(f'{output_dir}\\{outputFilename}', da.nansum(ds_stack,0), **profile)

# Accumulate Uncropped Raw Hourly .01h Grib files from EOL UCAR.
def accumulate(input_dir, output_dir):
    # Assumes Directory Name is the name of the storm and wanted name of the output files.
    # I.e: input_dir=r'Z:\LWI_StageIV\Hurricane Cindy 2005' ==> output file will be 'Hurricane Cindy 2005.tif'
    outputFilename = input_dir.split('\\')[-1]

    # get all files with extension .01h (the default pattern for uncompressed 1hr precip data from UCAR EOL).
    # Script is not accounting for potential naming patterns for AK and PR.
    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(input_dir),'*.01h'): 
        merge_dir.append((os.path.join(input_dir, filename)))

    merge_dir.sort()

    map2array=[]
    for raster in merge_dir:
        print(f'adding {raster} to dask array list: map2array')
        src = rasterio.open(raster)
        array = src.read(1)
        array[array==0] = np.nan    
        array[array==9999] = np.nan
        # map2array.append(read_raster(raster, band=1, block_size=10))
        map2array.append(da.from_array(array, chunks=array.shape))

    with rasterio.open(merge_dir[0]) as src:
        profile=src.profile

    ds_stack = da.stack(map2array)

    with rasterio.open(merge_dir[0]) as src:
        profile=src.profile
        profile.update(compress='lzw')
    print (f'writing Raster to {output_dir}\\{outputFilename}.tif')
    write_raster(f'{output_dir}\\{outputFilename}.tif', da.nansum(ds_stack,0), **profile)


def animate(input_dir, output_dir, crop_shp, dst_crs):
    outputFilename = input_dir.split('\\')[-1]+'.gif'
    img_dir = os.path.join(input_dir, 'img')
    projected_dir = os.path.join(input_dir, 'projected')
    # movieFilename = os.path.join(img_dir, outputFilename)

    sns.set_style("white")
    sns.set(font_scale=1.5)

    # Check whether the image and projection paths exists.
    # Create a new directory if not isExists.
    isExist_img = os.path.exists(img_dir)
    if not isExist_img:
        os.makedirs(img_dir)
    isExist_proj = os.path.exists(projected_dir)
    if not isExist_proj:
        os.makedirs(projected_dir)

    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(input_dir),'*.01h'): 
        merge_dir.append((os.path.join(input_dir, filename)))

    merge_dir.sort()

    arrayList = []
    print (f'\nwriting projected Raster files to {dst_crs} {projected_dir}')
    for raster in merge_dir:
        # print(f'adding {raster} to dask array list: map2array')
        src = rasterio.open(raster)
        projected_raster = projectRaster(raster, src, projected_dir)
        src_projected, kwargs = rasterio.open(projected_raster)
        cropped_raster, cropped_meta = es.crop_image(src_projected, crop_shp)
        # array = src.read(1)
        array = cropped_raster
        array[array==0] = np.nan    
        array[array==9999] = np.nan
        arrayList.append(array)
        src.close()
        src_projected.close()
    print (f'Writing plot images to {img_dir}')
    for i, array in enumerate(arrayList):
        # print (f'writing projected Raster file {projected_raster}')
        fig, ax = plt.subplots(figsize = (20, 10))
        im = ax.imshow(array.squeeze(), cmap='rainbow')
        ep.colorbar(im)
        title = merge_dir[i].split('\\')[-1]
        ax.set(title=f"{title} Hourly Precip (mm)")
        ax.set_axis_off()
        img_filename = img_dir + f'\\{title}.png'
        plt.savefig((img_filename))
        plt.close()

    print (f'Creating gif movie: {outputFilename}')
    imgList=[]
    for img_file in fnmatch.filter(os.listdir(img_dir),'*.png'): 
        imgList.append((os.path.join(img_dir, img_file)))
    imgList.sort()

    imagesArray = []
    for imageFile in imgList:
        # print(f'adding {imageFile} to imagesArray.')
        imagesArray.append(imageio.imread(imageFile))
    imageio.mimsave(os.path.join(output_dir, outputFilename), imagesArray, duration=0.1)
    print ('<----- Done. ----->\n')

# Animate Function inputs
# output_dir= r'Z:\LWI_StageIV\test\projected\iterate'
# crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
# dst_crs = 'EPSG:4326'
# img_dir = os.path.join(output_dir, 'img')
# stormName = projected_dir.split('\\')[-2]
# movieFilename = os.path.join(movie_dir, f'{stormName}-accum.gif')

def animateCumulative(input_dir, movie_dir):

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




# Build List of directories to make Cumulative Precip GeoTiffs for. Drop the first dir[1:]: '!Accumulated'
storm_dirs = next( os.walk(stormDir) )[1][1:]
for storm in storm_dirs:
    inputDir = os.path.join(stormDir, storm)
    # accumulate(inputDir,outputDir)
    # animate(inputDir,outputDir, crop_shp, dst_crs)
    # animateCumulative(inputDir, outputDir)
    