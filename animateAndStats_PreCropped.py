import rasterio
import os
import fnmatch
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import rasterstats

input_dir = r'Z:\LWI_StageIV\test'
output_dir= r'Z:\LWI_StageIV\!Accumulated'

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
        stats = rasterstats.zonal_stats(r"Z:\GIS\StageIv Boundary.shp", raster, stats=['min', 'max','mean', 'count', 'nodata'])
        percentMissing = stats[0]['nodata']/stats[0]['count']
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
    if (max > bigMax):
        bigMax = max
    # If over 3% of raster is Nan, count it as having missing data.
    if (rasterStats[raster][0]['percentMissing'] > 3.0):
        hasMissingHoursIntegerList.append(hour)
        percentAreaMissingList.append(rasterStats[raster][0]['percentMissing'])
hoursTotal = hour
hoursMissingTotal = len(hasMissingHoursIntegerList)
percentHoursMissing = round(hoursMissingTotal/hoursTotal, 2)* 100

# Animate
hour = 0
for i, array in enumerate(arrayList):
    hour += 1
    title = merge_dir[i].split('\\')[-1]
    img_filename = img_dir + f'\\{title}.png'
    crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
    fig, ax = plt.subplots(figsize=(20, 15))
    la = gpd.read_file("Z:\GIS\Louisiana.shp")
    bbox = la.total_bounds
    la_extent=bbox[[0,2,1,3]]
    bbox_lwi = crop_shp.total_bounds
    lwi_extent = bbox_lwi[[0,2,1,3]]
    array[array==9999] = np.nan
    im = plt.imshow(array, extent=lwi_extent, cmap='rainbow', vmin=0, vmax=bigMax)
    cb = plt.colorbar(im, shrink=.6)
    ax.set(title=f"{storm}\n{title} Incremental Precip (mm), Total Duration (hr): {hoursTotal} \nHour:{hour}, %Hours with Missing Data: {percentHoursMissing}%")
    ax.set_axis_off()
    la.boundary.plot(ax=plt.gca(), color='darkgrey')
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
