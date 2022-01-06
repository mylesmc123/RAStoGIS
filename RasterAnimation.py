import rasterio
import geopandas as gpd
# import dask.array as da
# from dask_rasterio import read_raster, write_raster
import os
import fnmatch
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthpy.spatial as es
import seaborn as sns
import matplotlib.pyplot as plt
import earthpy.plot as ep
import imageio



input_dir=r"Z:\LWI_StageIV\test"
crop_shp = gpd.read_file(r"Z:\GIS\StageIv Boundary.shp")
dst_crs = 'EPSG:4326'
def animate(input_dir, crop_shp, dst_crs):
    outputFilename = input_dir.split('\\')[-1]+'.gif'
    img_dir = os.path.join(input_dir, 'img')
    projected_dir = os.path.join(input_dir, 'projected')
    movieFilename = os.path.join(img_dir, outputFilename)

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
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        projected_raster = projected_dir + '\\' + raster.split("\\")[-1]
        with rasterio.open(projected_raster, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        src_projected = rasterio.open(projected_raster)
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
    imageio.mimsave(movieFilename, imagesArray, duration=0.5)
    print ('<----- Done. ----->\n')

animate(input_dir, crop_shp, dst_crs)