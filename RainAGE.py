import rasterio
from rasterio import mask as msk
import dask.array as da
from dask_rasterio import read_raster, write_raster
import os
import fnmatch
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from rasterio.warp import calculate_default_transform, reproject, Resampling
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import rasterstats


# Takes a directory of Stage IV precip data from UCAR (uncompressed) and accumulates .01h files to a single geotif.
# An image file of the geotif is also produced to quickly view the result.

# Initial Inputs
stormDir = r"P:\Projects\Office of Community Development\Working Files\Sensitivity Testing Oct2021\ST4_Gap_Analysis\NonTropicalStorms"
outputDir= os.path.join(stormDir, '!Accumulated-Amite')
# Shape that will be used to crop the raster.
crop_shp = gpd.read_file(r".\Amite\Amite_Basin_bbox.geojson")
# Shape that will be the Basin Map. Extents Must be within the crop_shp.
la_shp = gpd.read_file(r".\Amite\Amite_Basin_Outline.geojson")
# la_shp = gpd.read_file("Z:\GIS\Louisiana.shp")
dst_crs = 'EPSG:4326'

def projectRaster(raster_fn, raster_src, projected_dir, dst_crs):
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

# Deprecated
# def crop_raster(src, crop_shp):
    # cropped_raster, cropped_meta = es.crop_image(src, crop_shp)
    # array = cropped_raster
    # array[array==0] = np.nan    
    # array[array==9999] = np.nan
    # src.close()
    # return array

def cropRasterByMask(src_projected, crop_shp, cropped_raster_fn):
    geom = []
    coord = shapely.geometry.mapping(crop_shp)["features"][0]["geometry"]
    geom.append(coord)
    out_image, out_transform = msk.mask(src_projected, geom, crop=True)
    out_meta = src_projected.meta
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(os.path.join(cropped_raster_fn), "w", **out_meta) as dest:
        dest.write(out_image)

# Old Version. Deprecated.
# def create_image(array, title, output_dir, merge_dir, hours):
    # fig, ax = plt.subplots(figsize = (20, 10))
    # im = ax.imshow(array, cmap='rainbow', vmin=.1, vmax=400)
    # ep.colorbar(im)
    # # title = merge_dir[i].split('\\')[-1]
    # ax.set(title=f"{merge_dir[0]} to {title} Cumulative Precip (mm) {hours} Hours")
    # ax.set_axis_off()
    # img_filename = output_dir + f'\\{title}-accum.png'
    # plt.savefig((img_filename))
    # plt.close()

def create_image(array, title, img_dir, outputFilename, crop_shp, la_shp, bigMax):
    img_filename = os.path.join(img_dir, outputFilename)
    fig, ax = plt.subplots(figsize=(20, 15))
    
    # bbox = la_shp.total_bounds
    # la_extent=bbox[[0,2,1,3]]
    bbox_lwi = crop_shp.total_bounds
    lwi_extent = bbox_lwi[[0,2,1,3]]
    array[array==9999] = np.nan
    im = plt.imshow(array, extent=lwi_extent, cmap='rainbow', vmin=0, vmax=bigMax)
    cb = plt.colorbar(im, shrink=.6)
    ax.set(title=title)
    ax.set_axis_off()
    la_shp.boundary.plot(ax=plt.gca(), color='darkgrey')
    plt.savefig((img_filename+'.png'))
    plt.close()

# Writes out Projected hourly tifs, then writes Cropped hourly Tifs, Then Accumulates to a Single Tif.
def projectCrop(input_dir, output_dir, crop_shp, dst_crs):
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
    for filename in fnmatch.filter(os.listdir(input_dir),'*.grb2'): 
        merge_dir.append((os.path.join(input_dir, filename)))

    merge_dir.sort()

    # Get Shape of first array
    raster = merge_dir[0]
    src = rasterio.open(raster)
    projected_raster, kwargs = projectRaster(raster, src, projected_dir, dst_crs)
    src_projected = rasterio.open(projected_raster)
    # Write Cropped Raster to Disk
    cropped_raster_fn = cropped_dir + '\\' + raster.split("\\")[-1]
    cropRasterByMask(src_projected, crop_shp, cropped_raster_fn)
    cropped_src = rasterio.open(cropped_raster_fn)
    profile = cropped_src.profile
    array = cropped_src.read(1)
    array[array==0] = np.nan    
    array[array==9999] = np.nan
    firstArrayShape = array.shape
    src.close()
    src_projected.close()
    cropped_src.close()

    map2array=[]
    for raster in merge_dir:
        # print(f'projecting and cropping {raster}. Then adding to dask array list: map2array')
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
        # Check if Array Shape equal doesnt equal firstArrayShape and reshape it.
        if not (array.shape == firstArrayShape):
            #reshape
            array.resize(firstArrayShape, refcheck=False) 
            array[array==0] = np.nan

        # with rasterio.open(cropped_raster_fn, 'w', **kwargs) as dest:
        #     dest.write(array.squeeze().astype(rasterio.uint8), 1)
        #     profile = dest.profile
        # map2array.append(read_raster(raster, band=1, block_size=10))
        # map2array.append(da.from_array(array, chunks=500))
        src.close()
        src_projected.close()
        cropped_src.close()

    # ds_stack = da.stack(map2array)
    # print (f'writing Raster to {output_dir}\\{outputFilename}')
    # write_raster(f'{output_dir}\\{outputFilename}', da.nansum(ds_stack,0), **profile)

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

# Animates preCropped raster and provides missing data stats.
# Assumes the projectCrop function is run first.
def animateAndStats(input_dir, output_dir, la_shp, crop_shp):
    cropped_dir = os.path.join(input_dir, 'cropped')
    img_dir = os.path.join(cropped_dir, 'img')
    outputFilename = input_dir.split('\\')[-1]+'-Inc-Cropped.gif'
    storm = input_dir.split('\\')[-1]
    statsPlotOutput_fn = input_dir.split('\\')[-1]+'-stats.png'
    sns.set_style("white")
    sns.set(font_scale=1.5)

    isExist_img = os.path.exists(img_dir)
    if not isExist_img:
        os.makedirs(img_dir)

    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(cropped_dir),'*.01h'): 
        merge_dir.append((os.path.join(cropped_dir, filename)))
    for filename in fnmatch.filter(os.listdir(cropped_dir),'*.grb2'): 
        merge_dir.append((os.path.join(cropped_dir, filename)))

    merge_dir.sort()

    # Build Arrays for Animating and Statistics 
    arrayList = []
    rasterStats = {}
    for raster in merge_dir:
            # print(f'adding {raster} to dask array list: map2array')
            src = rasterio.open(raster)
            array = src.read(1)
            # array[array==0] = np.nan    
            array[array==9999] = np.nan
            arrayList.append(array)
            
            # Get Rasterstats
            stats = rasterstats.zonal_stats(crop_shp, raster, stats=['min', 'max','mean', 'count', 'nodata'])
            percentMissing = stats[0]['nodata']/(stats[0]['count'] + stats[0]['nodata'])
            percentMissing = round(percentMissing * 100, 2)
            percentMissing
            stats[0]['percentMissing'] = percentMissing
            rasterStats[raster] = stats
            src.close()

    # Get the biggest max for the plot colorBar maxValue to be.
    # Record for which Hours that have a missing data covering more than 3% of the raster area.
    # Make list of the percent Area missing for each Hour that has missing data
    hasMissingHoursIntegerList = []
    percentAreaMissingList = []
    bigMax = 0
    hour = 0
    for raster in rasterStats:
        hour +=1
        max = rasterStats[raster][0]['max']
        if max is not None and (max > bigMax):
            bigMax = max
        # If over 3% of raster is Nan, count it as having missing data.
        if (rasterStats[raster][0]['percentMissing'] > 3.0):
            hasMissingHoursIntegerList.append(hour)
            percentAreaMissingList.append(rasterStats[raster][0]['percentMissing'])
    hoursTotal = hour
    hoursMissingTotal = len(hasMissingHoursIntegerList)
    percentHoursMissing = round(hoursMissingTotal/hoursTotal, 2)* 100
    # summaryTableColumnKeys = ['Event', 'Total Duration (hr)', 'Hours With Missing Data', '%% of Duration With Missing Data' ]
    # statsTable = [  {'Event':storm},
    #                 {'Total Duration (hr)': hoursTotal},
    #                 {'Hours With Missing Data': hoursMissingTotal},
    #                 {'%% of Duration With Missing Data': percentHoursMissing}
    #              ]
    # statsTable = [{ 'Event':storm,
    #                 'Total Duration (hr)': hoursTotal,
    #                 'Hours With Missing Data': hoursMissingTotal,
    #                 '%% of Duration With Missing Data': percentHoursMissing
    # }]
    stats_df = pd.DataFrame()
    stats_df['Event'] = [storm]
    stats_df['Total Duration (hr)'] = [hoursTotal]
    stats_df['Hours With Missing Data'] = [hoursMissingTotal]
    stats_df['%% of Duration With Missing Data'] = [percentHoursMissing]
    # Animate
    hour = 0
    for i, array in enumerate(arrayList):
        hour += 1
        title = merge_dir[i].split('\\')[-1]
        img_filename = img_dir + f'\\{title}.png'
        # crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
        fig, ax = plt.subplots(figsize=(20, 15))
        # bbox = la_shp.total_bounds
        # la_extent=bbox[[0,2,1,3]]
        bbox_lwi = crop_shp.total_bounds
        lwi_extent = bbox_lwi[[0,2,1,3]]
        array[array==9999] = np.nan
        im = plt.imshow(array, extent=lwi_extent, cmap='rainbow', vmin=0, vmax=bigMax)
        cb = plt.colorbar(im, shrink=.6)
        ax.set(title=f"{storm}\n{title} Incremental Precip (mm), Total Duration (hr): {hoursTotal} \nHour:{hour}, %Hours with Missing Data: {percentHoursMissing}%")
        ax.set_axis_off()
        la_shp.boundary.plot(ax=plt.gca(), color='darkgrey')
        plt.savefig(img_filename)
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

    # Set up Data arrays for Stats Timeseries Plots
    x_arr = np.arange(1, hoursTotal)
    y_MissingData=[]
    y_PercentAreaMissing = []
    y=0
    for hr in x_arr:
        if hr in hasMissingHoursIntegerList:
            y+=1
            i = hasMissingHoursIntegerList.index(hr)  
            y_PercentAreaMissing.append(percentAreaMissingList[i])
        else:
            y_PercentAreaMissing.append(0)   
        y_MissingData.append(y)
        
    y_arr_MissingData = np.array(y_MissingData)
    y_arr_PercentAreaMissing = np.array(y_PercentAreaMissing)
    fig, ax = plt.subplots(figsize=(20, 10))

    # Hourly Datasets with Missing Values Plot
    plt.subplot(121)
    plt.suptitle(f'{storm}')
    plt.plot(x_arr,y_arr_MissingData)
    plt.ylabel('Hourly Datasets with Missing Values')
    plt.xlabel('Time (hr)')

    # Percent Area of Missing Data Plot
    plt.subplot(122)
    plt.plot(x_arr,y_arr_PercentAreaMissing)
    plt.ylabel('Percent Area Missing')
    plt.xlabel('Time (hr)')
    # plt.show()
    plt.savefig(os.path.join(output_dir, statsPlotOutput_fn))
    plt.close()
    return stats_df

# Animates and accumulates through time.
# Assumuming precropped raster.
def animateCumulative(input_dir, movie_dir, crop_shp, la_shp):

    # Animate Function inputs
    cropped_dir = os.path.join(input_dir, 'cropped' )
    output_dir = os.path.join(cropped_dir, 'iterate' )
    # crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
    # dst_crs = 'EPSG:4326'
    img_dir = os.path.join(output_dir, 'img')
    stormName = input_dir.split('\\')[-1]
    movieFilename = os.path.join(movie_dir, f'{stormName}-accum.gif')
    accumRaster = os.path.join(movie_dir, f'{stormName}-cropped.tif')
    
    #Get max accumulated cropped raster value.
    stats = rasterstats.zonal_stats(crop_shp, accumRaster, stats=['max'])
    bigMax = stats[0]['max']
    
    #  Get all files with extension .01h (the default pattern for uncompressed 1hr precip data from UCAR EOL).
    # Script is not accounting for potential naming patterns for AK and PR.
    merge_dir=[]
    for filename in fnmatch.filter(os.listdir(cropped_dir),'*.01h'): 
        merge_dir.append((os.path.join(cropped_dir, filename)))
    for filename in fnmatch.filter(os.listdir(cropped_dir),'*.grb2'): 
        merge_dir.append((os.path.join(cropped_dir, filename)))

    merge_dir.sort()

    # Get first raster profile
    with rasterio.open(merge_dir[0]) as src:
            profile=src.profile
    
    # Get first raster file name
    firstRasterTitle = str(merge_dir[0].split('\\')[-1])
    
    # Get Shape of first array
    cropped_raster_fn = cropped_dir + '\\' + firstRasterTitle
    cropped_src = rasterio.open(cropped_raster_fn)
    profile = cropped_src.profile
    array = cropped_src.read(1) 
    array[array==9999] = np.nan
    firstArrayShape = array.shape
    cropped_src.close()

    hours = 0
    map2array=[]
    for raster in merge_dir:
        hours += 1
        outputFilename = raster.split('\\')[-1]
        title = f"{stormName}\n{firstRasterTitle} to {outputFilename} Cumulative Precip (mm) {hours} (hr)"
        # print(f'adding {raster} to dask array list: map2array')
        src = rasterio.open(raster)
        array = src.read(1)
        # array = crop_raster(src, crop_shp)
        # array[array==0] = np.nan    
        array[array==9999] = np.nan
        # Check if Array Shape equal doesnt equal firstArrayShape and reshape it.
        if not (array.shape == firstArrayShape):
            #reshape
            array.resize(firstArrayShape, refcheck=False) 
            # array[array==0] = np.nan
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
            create_image(array, title, img_dir, outputFilename, crop_shp, la_shp, bigMax)
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
            create_image(array_accum, title, img_dir, outputFilename, crop_shp, la_shp, bigMax)
            # Empty out the Array
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

# Build Empty Summary Table DataFrame.
summary_df = pd.DataFrame()

# Main function calls.
# for storm in storm_dirs:
#     print (storm)
#     inputDir = os.path.join(stormDir, storm)
#     projectCrop(inputDir, outputDir, crop_shp, dst_crs)
    # stats_df = animateAndStats(inputDir, outputDir, la_shp, crop_shp)
    # summary_df = summary_df.append(stats_df)
    # animateCumulative(inputDir, outputDir, crop_shp, la_shp)for storm in storm_dirs:

for storm in storm_dirs:
    print (f'stats for {storm}')
    inputDir = os.path.join(stormDir, storm)
    # projectCrop(inputDir, outputDir, crop_shp, dst_crs)
    stats_df = animateAndStats(inputDir, outputDir, la_shp, crop_shp)
    summary_df = summary_df.append(stats_df)
    stats_df.to_csv(os.path.join(outputDir, 'SummaryStatsAppendable.csv'), mode='a',index=False, header=True)
    # animateCumulative(inputDir, outputDir, crop_shp, la_shp)
# Export Summary Table to CSV.
summary_df.to_csv(os.path.join(outputDir, 'SummaryStats.csv'), index=False, header=True)