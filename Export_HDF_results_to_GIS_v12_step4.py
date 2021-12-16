#*************************************************************************************
#-------This script exports 2D Flow area polygons & WSE results for each Timestep
#-------Cross-section (XS) polylines and WSE results are also exported to shapefile
#-------HDF format must be developed from Hec-RAS 6.0 Beta 3---------------------


import os
import shutil

import numpy as np
import h5py
import shapefile
import time
from osgeo import ogr, gdal, osr
import shapefile

#-------Locations of HDF file and export directory

# model_dir = r'C:\Users\dgilles\Desktop\MRS_Forecast_System\20210422_Model_Future_Cond'
model_dir = r"Z:\Amite\RAS_6storms_V1_longBarry"
hdf_filename = os.path.join(model_dir, "Amite_20200114.p13.hdf")



curr_date = time.strftime("%Y%m%d_%H")
# postp_dir = r'C:\Users\dgilles\Desktop\MRS_Forecast_System\post_process_GIS'
postp_dir = r'.\post_process_GIS'
home_dir = os.path.join(postp_dir,str(curr_date))
tempDir = os.path.join(home_dir,"tempfiles")

# postp_area = r'C:\Users\dgilles\Desktop\MRS_Forecast_System\postp_area.shp'
postp_area = os.path.join(postp_dir, "postp_area.shp")



if not os.path.exists(home_dir):
    os.makedirs(home_dir)


#-------------Delete Temp Directory--------
def tempDirSweep(tempDir):
    if not os.path.exists(tempDir):
        os.makedirs(tempDir)

    for f in os.listdir(tempDir):
        file_object_path = os.path.join(tempDir, f)
        if os.path.isfile(file_object_path):
            os.unlink(file_object_path)
        else:
            shutil.rmtree(file_object_path)
    return None
#------------------------------------------



#******Create/Sweep temp folder***
tempDirSweep(tempDir)
#***********************************

# Simply gets the names of the 2D Flow Areas in the Plan's geometry
def get2DAreaNames(hf):
    hdf2DFlow = hf['Results']['Unsteady']['Geometry Info']['2D Area(s)']
    AreaNames = []
    for key in hdf2DFlow:
        if key in ['Cell Spacing', "Manning's n", 'Names', 'Tolerances']:
            continue
        else:
            AreaNames.append(key)  # List of 2D Area names
    return AreaNames


def get2DArea_cellcenter_pts(curr_2DArea):
    hdf2DFlow_geo = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DFlow_geo[curr_2DArea]['Cells Center Coordinate']
    data_list = np.zeros((2,), dtype='float64')
    data_list = np.array(dataset).tolist()
    # print(data_list)
    return data_list


def get2DCells_min_elev(curr_2DArea):
    hdf2DFlow_geo = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DFlow_geo[curr_2DArea]['Cells Minimum Elevation']
    # print (dataset)
    #data_list = np.zeros([1, ], dtype='float64')
    data_list = np.array(dataset).tolist()
    return data_list


def get2DArea_wse_data(curr_2DArea):
    hdf2DFlow_wse_data = hf['Results']['Unsteady']['Output']['Output Blocks'] \
        ['Base Output']['Unsteady Time Series']['2D Flow Areas']

    dataset = hdf2DFlow_wse_data[curr_2DArea]['Water Surface']

    # data_list = np.zeros((timesteps,), dtype='float64')
    # dataset.read_direct(data_list, np.s_[0:timesteps,], np.s_[0:timesteps])
    data_list = np.array(dataset).tolist()

    return data_list

def get_timesteps(hf):
    hdf_timesteps = hf['Results']['Unsteady']['Output']['Output Blocks']['Base Output'] \
        ['Unsteady Time Series']['Time']
    timesteps = hdf_timesteps.shape[0]
    return timesteps


def getXSAttributes(hf):
    hdfXSAttributes = hf['Geometry']['Cross Sections']['Attributes']
    XSAttributes = []
    for key in hdfXSAttributes:
        if key in ['Cell Spacing', "Manning's n", 'Names', 'Tolerances']:
            continue
        else:
            XSAttributes.append(key)  # List of 2D Area names
    return XSAttributes

def get_FacePoints_Coordinates(hf, curr_2DArea):
    hdf2DFacePoints_Coordinates = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DFacePoints_Coordinates[curr_2DArea]['FacePoints Coordinate']
    data_list = np.array(dataset).tolist()
    return data_list

def get_Cells_Face_Info(hf, curr_2DArea):
    hdf2DCell_Face_Info = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell_Face_Info[curr_2DArea]['Cells Face and Orientation Info']
    data_list = np.array(dataset).tolist()
    return data_list

def get_Cells_FacePoints_Index(hf, curr_2DArea):
    hdf2DCell_Face_Index = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell_Face_Index[curr_2DArea]['Cells FacePoint Indexes']
    data_list = np.array(dataset).tolist()
    return data_list

def is_FacePoint_perimeter(hf, curr_2DArea):
    hdf2DCell_Face_is_perimeter = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell_Face_is_perimeter[curr_2DArea]['FacePoints Is Perimeter']
    data_list = np.array(dataset).tolist()
    return data_list

def get_faces_FacePoint_Index (hf, curr_2DArea):
    hdf2DCell = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell[curr_2DArea]['Faces FacePoint Indexes']
    data_list = np.array(dataset).tolist()
    return data_list

def get_faces_Perimeter_Info (hf, curr_2DArea):
    hdf2DCell = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell[curr_2DArea]['Faces Perimeter Info']
    data_list = np.array(dataset).tolist()
    return data_list

def get_faces_Perimeter_Values (hf, curr_2DArea):
    hdf2DCell = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell[curr_2DArea]['Faces Perimeter Values']
    data_list = np.array(dataset).tolist()
    return data_list

def get_face_orientation_info (hf, curr_2DArea):
    hdf2DCell = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell[curr_2DArea]['Cells Face and Orientation Info']
    data_list = np.array(dataset).tolist()
    return data_list

def get_face_orientation_values (hf, curr_2DArea):
    hdf2DCell = hf['Geometry']['2D Flow Areas']
    dataset = hdf2DCell[curr_2DArea]['Cells Face and Orientation Values']
    data_list = np.array(dataset).tolist()
    return data_list

#***************************Open HDF5 file and get 2DArea Names, Timestep
hf = h5py.File(hdf_filename, 'r')
list_of_2DAreas = get2DAreaNames(hf)
timesteps = get_timesteps(hf)
all_data = np.empty((0, timesteps + 6), ) # Includes an extra wse column for maximum row value

#Initialize shapefile of all 2D flow cells
poly_wse_shp = os.path.join(tempDir, 'test_polygon_all_data.shp')
w = shapefile.Writer(poly_wse_shp)
# ***********Begin writing to polygon shapefile using rows of cell index pts
# w = shapefile.Writer('test_polygon_all_data')   <-------- Currently  initialize outside of 2DArea loop
# Start writing to Shapefile
# Writing field names and types
# "C": Characters, text.
# "N": Numbers, with or without decimals.
# "F": Floats(same as "N").
# "L": Logical, for boolean True / False values.
# "D": Dates.
# "M": Memo
w.field('Area2D', 'C')
w.field('Cell_Index', 'N')
w.field('Easting', 'N', decimal=2)
w.field('Northing', 'N', decimal=2)
w.field('min_elev', 'N', decimal=2)

#Creating Results fields, same number as timesteps
i = 0
while i < timesteps:
    w.field('wse_' + str(i), 'N', decimal=2)
    i += 1
# Shapefile is closed after 2DArea loop

#Add a wse for maximum water surface at end
w.field('wse_max', 'N', decimal=2)

coord_sys = 'PROJCS["NAD_1983_BLM_Zone_15N_ftUS",GEOGCS' \
                '["GCS_North_American_1983",DATUM["D_North_American_1983",'\
                'SPHEROID["GRS_1980",6378137.0,298.257222101]],' \
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],' \
                'PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1640416.666666667],' \
                'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-93.0],'\
                'PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],'\
                'UNIT["Foot_US",0.3048006096012192]]'



#Loop through all 2D flow areas in HDF file and extract geometry pts and results
for curr_2DArea in list_of_2DAreas:
    print("Current 2D Area is: %s" % curr_2DArea)

    xy_pts = np.array(get2DArea_cellcenter_pts(curr_2DArea))
    min_elev = np.array(get2DCells_min_elev(curr_2DArea)).round(decimals=2)
    # transpose_min_elev = min_elev.T

    wse_data = np.array(get2DArea_wse_data(curr_2DArea))
    transpose_wse = wse_data.T.round(decimals=2)

    #Find WSE values that are equal to cell min elev, set to NaN, all others set to 1
    repeats_cell_min_elev = np.tile(min_elev, (timesteps,1)).T
    cell_depths = transpose_wse - repeats_cell_min_elev
    cell_depths[cell_depths > 0] = 1
    cell_depths[cell_depths == 0] = 0

    #Remove zero depth values
    filtered_transpose_wse = cell_depths * transpose_wse
    filtered_transpose_wse[filtered_transpose_wse==0] = -9999
    filtered_transpose_wse.round(decimals=2)
    max_of_row = np.max(filtered_transpose_wse, axis=1)




    cell_index = np.arange(xy_pts.shape[0])
    curr_2DArea_index = [curr_2DArea.decode('UTF-8')]* (xy_pts.shape[0])

    #Adding columns to results array
    all_data_for_curr_2DArea = np.column_stack((curr_2DArea_index,cell_index, xy_pts, min_elev))
    all_data_for_curr_2DArea = np.concatenate((all_data_for_curr_2DArea, filtered_transpose_wse), axis=1)
    all_data_for_curr_2DArea = np.column_stack((all_data_for_curr_2DArea, max_of_row))



    #Save into the overall dataset
    all_data = np.append(all_data, all_data_for_curr_2DArea, axis=0)

    # Assemble 2D Cell Polygons
    cell_face_info = get_Cells_Face_Info(hf, curr_2DArea)
    cell_face_xy_pts = get_FacePoints_Coordinates(hf, curr_2DArea)
    cell_face_index_pts = get_Cells_FacePoints_Index(hf, curr_2DArea)

    #Assemble info about perimeter faces and facepoints
    cell_facept_is_perimeter = is_FacePoint_perimeter(hf, curr_2DArea)
    face_facept_index = get_faces_FacePoint_Index(hf, curr_2DArea)
    face_perimeter_info = get_faces_Perimeter_Info(hf, curr_2DArea)
    face_perimeter_values = get_faces_Perimeter_Values(hf, curr_2DArea)
    face_orientation_info = get_face_orientation_info(hf, curr_2DArea)
    face_orientation_values = get_face_orientation_values(hf, curr_2DArea)


    #Assemble current polygons
    cell_ids = []
    index_size = len(cell_face_index_pts[0])
    curr_2DArea_Polygon_xy_pts = []



    cell_id = 0
    cell_ids = []
    for row in cell_face_index_pts:

        #find if facepoints are perimeter
        perimeter_facepts = []
        for facept in row:

            if facept != -1:

                if cell_facept_is_perimeter[facept] == -1:
                    perimeter_facepts.append(facept)
        #print(perimeter_facepts)
        #Declare empty polygon list for 2D cell
        polygon = []

        i = 0
        while i < index_size:

            curr_facept = row[i]

            if curr_facept != -1:
                polygon.append(cell_face_xy_pts[curr_facept])

            if i < (index_size -1) :
                next_facept = row[i+1]

            if i == (index_size -1):
                next_facept = row[0]


            if curr_facept in perimeter_facepts:
                if next_facept in perimeter_facepts:
                    face_index=0
                    for face in face_facept_index:

                        if curr_facept == face_facept_index[face_index][0]:
                            potential_face = face_index
                            if next_facept == face_facept_index[potential_face][1]:
                                next_is_first = False
                                curr_face_index = face_index
                                print("found face")
                                break

                        if next_facept == face_facept_index[face_index][0]:
                            potential_face = face_index
                            if curr_facept == face_facept_index[potential_face][1]:
                                next_is_first = True
                                curr_face_index = face_index
                                print("found face")
                                break

                        face_index +=1


                    perimeter_st_pt = face_perimeter_info[curr_face_index][0]
                    num_perimeter_pts = face_perimeter_info[curr_face_index][1]
                    perimeter_end_pt = perimeter_st_pt + num_perimeter_pts - 1
                    perimeter_pt_index = perimeter_st_pt

                    extra_perimeter_xy_pts = []

                    print("...adding perimeter pts, for face %s" % curr_face_index )
                    while perimeter_pt_index <= perimeter_end_pt:
                        # polygon.append(face_perimeter_values[perimeter_pt_index])
                        extra_perimeter_xy_pts.append(face_perimeter_values[perimeter_pt_index])
                        perimeter_pt_index += 1

                    if next_is_first:
                        extra_perimeter_xy_pts = extra_perimeter_xy_pts[::-1]

                    polygon.extend(extra_perimeter_xy_pts)
                #polygon.append(cell_face_xy_pts[next_facept])

            i += 1

        #Append the first face pt coordinate
        polygon.append(cell_face_xy_pts[row[0]])

        #Append to the total 2D Area set if more than 2 points (there are some lateral weirs represented like this)
        if sum(1 for n in row if n != -1)>=3:
            curr_2DArea_Polygon_xy_pts.append(polygon)
            #Keep track of cell_ids that make it into the polygon set
            cell_ids.append(cell_id)

        cell_id += 1





    #--------------Saving polygons and records to shapefile-------------
    print ("writing %s polygons to shapefile..." %curr_2DArea.decode('UTF-8'))
    str_curr_2DArea = curr_2DArea.decode('UTF-8')
    for row_id, poly_row in enumerate(curr_2DArea_Polygon_xy_pts):
        if len(poly_row) > 2:
            w.poly([poly_row[::-1]]) #clockwise flip
            #w.record(INT=nr, LOWPREC=nr, MEDPREC=nr, HIGH)
            #w.record(Area2D=str_curr_2DArea,Cell_Index=cell_id, Easting=all_data_for_curr_2DArea[cell_id][])
            #w.record('Area2D', str_curr_2DArea)
            #w.record('Cell Index', cell_id)
            records = np.array(all_data_for_curr_2DArea[cell_ids[row_id]]).tolist()
            w.record(*records)


#Close 2DArea polygon shapefile
print("Closing shapefile with all 2D Area polygons.")
w.close()

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_polygon_all_data.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#********************************Buffer to fix self-intersections*********************************
def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None

    inputds = None
    inputlyr = None

#***************************************Fix self intersections*********************************************
print('Buffering 2D Area polygon shapefile by small amount to remove self-intersections and slivers...')

poly_wse_shp_buffer = os.path.join(tempDir, 'test_polygon_all_data_buffer.shp')
createBuffer(poly_wse_shp, poly_wse_shp_buffer, 5)

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_polygon_all_data_buffer.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#*******************************Dissolve final Polygon*******************************************
print("Dissolving 2D buffered polygons...")

def dissolve_polygon(input_shp, out_file):
    ds = ogr.Open(input_shp, 0)
    layer = ds.GetLayer()
    print(layer.GetGeomType())
    # -> polygons
    # empty geometry
    union_poly = ogr.Geometry(ogr.wkbPolygon)
    # make the union of polygons
    for feature in layer:
        geom = feature.GetGeometryRef()
        union_poly = union_poly.Union(geom)

    # print (union_poly)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = driver.CreateDataSource(out_file)
    srs = layer.GetSpatialRef()
    outLayer = outDataSource.CreateLayer('', srs, ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(union_poly)

    outLayer.CreateFeature(outFeature)

    outFeature = None
    OutLayer = None
    OutDataSource = None



poly_wse_shp_buffer_dissolved = os.path.join(tempDir, "test_polygon_all_data_buffer_dissolved.shp")
dissolve_polygon(poly_wse_shp_buffer, poly_wse_shp_buffer_dissolved)

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir, "test_polygon_all_data_buffer_dissolved" + '.prj'), 'w') as f:
    f.write(coord_sys)
    f.close


#*************************************************************************************************
#-----------------------------Loop through and get XS Polylines and Results-----------------------
#*************************************************************************************************
def get_XS_names (hf):
    try:
        dfXS_geo = hf['Geometry']['Cross Sections']['Attributes']
        dataset = dfXS_geo
        # print (dataset)
        # data_list = np.zeros([1, ], dtype='float64')
        data_list = np.array(dataset).tolist()
        print ("hf['Geometry']['Cross Sections']['Attributes']" + " does not exist")
        return data_list
    except:
        return None

def get_XS_polyline_info (hf):
    try:
        dfXS_geo = hf['Geometry']['Cross Sections']['Polyline Info']
        dataset = dfXS_geo
        # print (dataset)
        # data_list = np.zeros([1, ], dtype='float64')
        data_list = np.array(dataset).tolist()
        print ("hf['Geometry']['Cross Sections']['Polyline Info']" + " does not exist")
        return data_list
    except:
        return None

def get_XS_polyline_points (hf):
    try:
        dfXS_geo = hf['Geometry']['Cross Sections']['Polyline Points']
        dataset = dfXS_geo
        # print (dataset)
        # data_list = np.zeros([1, ], dtype='float64')
        data_list = np.array(dataset).tolist()
        print ("hf['Geometry']['Cross Sections']['Polyline Points']" + " does not exist")
        return data_list
    except:
        return None

def get_XS_wse_results (hf):
    try:
        dfXS_results = hf['Results']['Unsteady']['Output']['Output Blocks'] \
            ['Base Output']['Unsteady Time Series']['Cross Sections']['Water Surface']
        dataset = dfXS_results
        # print (dataset)
        # data_list = np.zeros([1, ], dtype='float64')
        data_list = np.array(dataset).tolist()
        print ("hf['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['Cross Sections']['Water Surface']" + " does not exist")
        return data_list
    except:
        return None

#-------------------------------------------------------------------------------------
#--------------------Get XS data from HDF file------------------------------------------
#*******************************************************************************************
print("Accessing XS Data from HDF file...")

XS_names = get_XS_names(hf)
XS_polyline_info = get_XS_polyline_info(hf)
XS_polyline_pts = get_XS_polyline_points(hf)
XS_wse_results = (np.array(get_XS_wse_results(hf))).T.round(decimals=2)

max_of_XS_row = np.max(XS_wse_results, axis=1)
XS_wse_results = np.column_stack((XS_wse_results, max_of_XS_row))

#-----------------Start assembling XS polyline point collections---------------------------
print('Beginning to assemble XS Polyline point collections...')
XS_polyline_pts_collections = []
for XS_index, XS in enumerate(XS_names):
    st_pt = XS_polyline_info[XS_index][0]
    num_pts = XS_polyline_info[XS_index][1]
    polyline =[]
    i = 0
    while i < num_pts:
        polyline.append(XS_polyline_pts[st_pt + i])
        i += 1
    XS_polyline_pts_collections.append(polyline)



#*****************Initialize Output XS Polyline Shapefile*******************************
#Initialize shapefile of all 2D flow cells
w = shapefile.Writer(os.path.join(tempDir,'test_XS_results_all_data'))


# Writing field names and types
# "C": Characters, text.
# "N": Numbers, with or without decimals.
# "F": Floats(same as "N").
# "L": Logical, for boolean True / False values.
# "D": Dates.
# "M": Memo
w.field('River', 'C')
w.field('Reach','C')
w.field('Station', 'C')
#Creating Results fields, same number as timesteps
i = 0
while i < timesteps:
    w.field('wse_' + str(i), 'N', decimal=2)
    i += 1

#Add a wse for maximum water surface at end
w.field('wse_max', 'N', decimal=2)


for XS_index, XS in enumerate(XS_names):

    w.line([XS_polyline_pts_collections[XS_index]])

    results_records = XS_wse_results[XS_index].tolist()
    results_records.insert(0, XS_names[XS_index][0].decode('UTF-8'))
    results_records.insert(1, XS_names[XS_index][1].decode('UTF-8'))
    results_records.insert(2, XS_names[XS_index][2].decode('UTF-8'))
    w.record(*results_records)

coord_sys = 'PROJCS["NAD_1983_BLM_Zone_15N_ftUS",GEOGCS' \
                '["GCS_North_American_1983",DATUM["D_North_American_1983",'\
                'SPHEROID["GRS_1980",6378137.0,298.257222101]],' \
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],' \
                'PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1640416.666666667],' \
                'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-93.0],'\
                'PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],'\
                'UNIT["Foot_US",0.3048006096012192]]'

#Close 2DArea polygon shapefile
print("Closing shapefile with all XS results.")
w.close()

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_XS_results_all_data.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#*************************************************************************************************
#-----------------------------Loop through and get Centerlines------------------------------------
#*************************************************************************************************
def get_stream_names (hf):
    dfstream_names = hf['Geometry']['River Centerlines']['Attributes']
    dataset = dfstream_names
    # print (dataset)
    # data_list = np.zeros([1, ], dtype='float64')
    data_list = np.array(dataset).tolist()
    return data_list

def get_stream_polyline_info (hf):
    dfstream_geo_info = hf['Geometry']['River Centerlines']['Polyline Info']
    dataset = dfstream_geo_info
    # print (dataset)
    # data_list = np.zeros([1, ], dtype='float64')
    data_list = np.array(dataset).tolist()
    return data_list

def get_stream_polyline_points (hf):
    dfstream_geo_pts = hf['Geometry']['River Centerlines']['Polyline Points']
    dataset = dfstream_geo_pts
    # print (dataset)
    # data_list = np.zeros([1, ], dtype='float64')
    data_list = np.array(dataset).tolist()
    return data_list
#-------------------------------------------------------------------------------------
#--------------------Get Stream Centerline data from HDF file------------------------------------------
#*******************************************************************************************
print("Accessing XS Data from HDF file...")

stream_names = get_stream_names(hf)
stream_polyline_info = get_stream_polyline_info(hf)
stream_polyline_pts = get_stream_polyline_points(hf)


#-----------------Start assembling Stream Centerline polyline point collections---------------------------
print('Beginning to assemble Stream Centerline point collections...')
stream_polyline_pts_collections = []
for stream_index, stream in enumerate(stream_names):
    st_pt = stream_polyline_info[stream_index][0]
    num_pts = stream_polyline_info[stream_index][1]
    polyline =[]
    i = 0
    while i < num_pts:
        polyline.append(stream_polyline_pts[st_pt + i])
        i += 1
    stream_polyline_pts_collections.append(polyline)
#*****************Initialize Output Stream Polyline Shapefile*******************************
#Initialize shapefile of Stream centerlines
w = shapefile.Writer(os.path.join(home_dir,'test_stream_centerlines'))

# Writing field names and types
# "C": Characters, text.
# "N": Numbers, with or without decimals.
# "F": Floats(same as "N").
# "L": Logical, for boolean True / False values.
# "D": Dates.
# "M": Memo
w.field('River', 'C', 16)
w.field('Reach','C')
#w.field('Station', 'C')
#Creating Results fields, same number as timesteps
#i = 0
#while i < timesteps:
    #w.field('wse_' + str(i), 'N', decimal=2)
    #i += 1

for stream_index, stream in enumerate(stream_names):

    w.line([stream_polyline_pts_collections[stream_index]])

    #results_records = XS_wse_results[stream_polyline_pts_collections].tolist()
    results_records = stream_names[stream_index][0:2]
    #results_records.insert(1, XS_names[stream_polyline_pts_collections][1].decode('UTF-8'))
    #results_records.insert(2, XS_names[stream_polyline_pts_collections][2].decode('UTF-8'))
    w.record(*results_records)

coord_sys = 'PROJCS["NAD_1983_BLM_Zone_15N_ftUS",GEOGCS' \
                '["GCS_North_American_1983",DATUM["D_North_American_1983",'\
                'SPHEROID["GRS_1980",6378137.0,298.257222101]],' \
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],' \
                'PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1640416.666666667],' \
                'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-93.0],'\
                'PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],'\
                'UNIT["Foot_US",0.3048006096012192]]'

#Close Stream Centerline shapefile
print("Closing shapefile with all Stream Centerlines.")
w.close()

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(home_dir,'test_stream_centerlines.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#-----------------Start assembling Boundary polygon point collections---------------------------

print('Beginning to assemble Stream boundary polygon collections...')

#Remove duplicate river names from list
river_names = []
for stream in stream_names:
    if stream[0] not in river_names:
        river_names.append(stream[0])

boundary_polygon_collections = []

for river in river_names:
    print("Assembling boundary polygon for %s" %str(river))
    left_side_pts = []
    right_side_pts = []
    curr_stream_polygon = []
    curr_stream_XS_indexes = []

    #Assemble list of cross-sections indexes for current stream
    for XS_index, XS in enumerate(XS_names):
        if XS[0] == river:
            curr_stream_XS_indexes.append(XS_index)

    #Farthest upstream XS pts
    upstream_XS_index = curr_stream_XS_indexes[0]
    st_pt = XS_polyline_info[upstream_XS_index][0]
    num_pts = XS_polyline_info[upstream_XS_index][1]
    upstream_polyline_pts = []
    i = 0
    while i < num_pts:
        #Assembling from right to left
        upstream_polyline_pts.insert(0, XS_polyline_pts[st_pt + i])
        i += 1

    #Farthest downstream XS pts
    downstream_XS_index = curr_stream_XS_indexes[-1]
    st_pt = XS_polyline_info[downstream_XS_index][0]
    num_pts = XS_polyline_info[downstream_XS_index][1]
    downstream_polyline_pts = []
    i = 0
    while i < num_pts:
        # Assembling from left to right
        downstream_polyline_pts.append(XS_polyline_pts[st_pt + i])
        i += 1

    #Iterate through remaining list of cross-sections, skipping first and last XSs
    for XS_index in curr_stream_XS_indexes[1:-1]:
        left_side_index = XS_polyline_info[XS_index][0]
        right_side_index = left_side_index + (XS_polyline_info[XS_index][1] - 1)

        left_side_pts.append(XS_polyline_pts[left_side_index])
        right_side_pts.insert(0, XS_polyline_pts[right_side_index])

    #Assemble polygon sides starting with upstream, left side, downstream, right side
    curr_stream_polygon = upstream_polyline_pts + left_side_pts + downstream_polyline_pts + right_side_pts

    #Add current stream boundary polygon to collection
    boundary_polygon_collections.append(curr_stream_polygon)



#*****************Initialize Output Boundary Polygons Shapefile*******************************
#Initialize shapefile of boundary Polygons
print('Beginning to create Stream boundary polygon shapefile...')
raw_boundary = os.path.join(tempDir,'raw_boundary_polygons.shp')
w = shapefile.Writer(raw_boundary)


# Writing field names and types
# "C": Characters, text.
# "N": Numbers, with or without decimals.
# "F": Floats(same as "N").
# "L": Logical, for boolean True / False values.
# "D": Dates.
# "M": Memo
w.field('River', 'C', )
#w.field('Reach','C')
#w.field('Station', 'C')
#Creating Results fields, same number as timesteps

for river_index, river in enumerate(river_names):
    w.poly([boundary_polygon_collections[river_index]])
    results_records = river.decode('UTF-8')
    w.record(*results_records)

coord_sys = 'PROJCS["NAD_1983_BLM_Zone_15N_ftUS",GEOGCS' \
                '["GCS_North_American_1983",DATUM["D_North_American_1983",'\
                'SPHEROID["GRS_1980",6378137.0,298.257222101]],' \
                'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],' \
                'PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",1640416.666666667],' \
                'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-93.0],'\
                'PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],'\
                'UNIT["Foot_US",0.3048006096012192]]'

#Close boundary polygon shapefile
print("Closing shapefile with all Boundary Polygons.")
w.close()

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'raw_boundary_polygons.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#***************************************Fix self intersections*********************************************
print('Buffering boundary polygon shapefile by small amount to remove self-intersections and slivers...')
boundary_buffer = os.path.join(tempDir, "test_boundary_polygon_temp_buffer" + ".shp")
createBuffer(raw_boundary,boundary_buffer, 50)

#***************************************Buffer Stream Centerlines******************************************
print('Buffering centerline shapefile by 100ft...')
test_stream_centerlines = os.path.join(home_dir,'test_stream_centerlines.shp')
test_stream_centerlines_buffer = os.path.join(tempDir,'test_stream_centerlines_buffer.shp')
createBuffer(test_stream_centerlines, test_stream_centerlines_buffer, 100)

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_stream_centerlines_buffer.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#***********************************Merging buffered centerlines w/ boundary polygon***************************
print("Merging buffered centerlines w/ boundary polygon...")
def merge(target, source):
     # layer of target shp
     driver = ogr.GetDriverByName("ESRI Shapefile")
     ds_t = driver.Open(target, 1)
     tr_layer = ds_t.GetLayer()

     # layer of soruce shp
     driver = ogr.GetDriverByName("ESRI Shapefile")
     ds_s = driver.Open(source, 1)
     sr_layer = ds_s.GetLayer()

     # copy features:
     for f in sr_layer:
         defn = tr_layer.GetLayerDefn()
         out_feat = ogr.Feature(defn)

         for i in range(0, defn.GetFieldCount()):
             out_feat.SetField(defn.GetFieldDefn(i).GetNameRef(), f.GetField(i))

         out_feat.SetGeometry(f.GetGeometryRef().Clone())
         tr_layer.CreateFeature(out_feat)
         out_feat = None

merge(boundary_buffer, test_stream_centerlines_buffer)

#***********************************Erase 1D area by 2D polygons*****************************************
print('Erasing the 2D polygon buffered areas from the boundary polygon...')
def erase_shapes(to_erase, eraser, out_file):
    feat1 = ogr.Open(to_erase,1)
    feat2 = ogr.Open(eraser,1)
    feat1Layer = feat1.GetLayer()
    feat2Layer = feat2.GetLayer()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = driver.CreateDataSource(out_file)
    srs = feat1Layer.GetSpatialRef()
    outLayer = outDataSource.CreateLayer('', srs, ogr.wkbPolygon)
    out_ds = feat1Layer.Erase(feat2Layer, outLayer)
    out_ds = None

final_1D_boundary = os.path.join(tempDir,'test_boundary_polygon_erase_2DArea.shp')
erase_shapes(boundary_buffer, poly_wse_shp_buffer_dissolved,final_1D_boundary)

print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_boundary_polygon_erase_2DArea.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()


#**********************************Multipart to singlepart*****************************
print("Converting Boundary Polygon from Multipart to Singlepart...")
def multipoly2poly(in_lyr, out_lyr):
    for in_feat in in_lyr:
        geom = in_feat.GetGeometryRef()
        if geom.GetGeometryName() == 'MULTIPOLYGON':
            for geom_part in geom:
                addPolygon(geom_part.ExportToWkb(), out_lyr)
        else:
            addPolygon(geom.ExportToWkb(), out_lyr)

def addPolygon(simplePolygon, out_lyr):
    featureDefn = out_lyr.GetLayerDefn()
    polygon = ogr.CreateGeometryFromWkb(simplePolygon)
    out_feat = ogr.Feature(featureDefn)
    out_feat.SetGeometry(polygon)
    out_lyr.CreateFeature(out_feat)
    print ('Polygon added.')

from osgeo import gdal
gdal.UseExceptions()
driver = ogr.GetDriverByName('ESRI Shapefile')
in_ds = driver.Open(final_1D_boundary, 0)
in_lyr = in_ds.GetLayer()
final_1D_boundary_singlepart = os.path.join(tempDir,'test_boundary_polygon_erase_2DArea_singlepart.shp')

if os.path.exists(final_1D_boundary_singlepart):
    driver.DeleteDataSource(final_1D_boundary_singlepart)
out_ds = driver.CreateDataSource(final_1D_boundary_singlepart)
out_lyr = out_ds.CreateLayer('poly', geom_type=ogr.wkbPolygon)
multipoly2poly(in_lyr, out_lyr)

#**********************************Calculate and remove small polygons****************
print("Calculating and removing small polygons...")
new_field = ogr.FieldDefn('Area', ogr.OFTInteger)
new_field.SetWidth(12)
out_lyr.CreateField(new_field)

for feature in out_lyr:
    try:
        geom = feature.GetGeometryRef()
        area = geom.GetArea() / (43560)  # (convert to acres)
        feature.SetField("Area", area)
        out_lyr.SetFeature(feature)
    except RuntimeError as err:
        print("Error in reading feature %d - %s" % (fid, err))
        continue  # skip this feature


print('...selecting polygons smaller than 20 acre...')
out_lyr.SetAttributeFilter("Area < 20")

for feature in out_lyr:
    feature.GetField("Area")
    out_lyr.DeleteFeature(feature.GetFID())
    in_ds.ExecuteSQL('REPACK ' + out_lyr.GetName())
out_lyr = None
in_ds = None
in_lyr = None
print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_boundary_polygon_erase_2DArea_singlepart.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

#**********************************Dissolve polygon************************************
print("Dissolving part of Boundary Polygon...")
def dissolve_polygon(input_shp, out_file):
    ds = ogr.Open(input_shp)
    layer = ds.GetLayer()
    print (layer.GetGeomType())
    # -> polygons
    # empty geometry
    union_poly = ogr.Geometry(ogr.wkbPolygon)
    # make the union of polygons
    for feature in layer:
          geom =feature.GetGeometryRef()
          union_poly = union_poly.Union(geom)

    #print (union_poly)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = driver.CreateDataSource(out_file)
    srs = layer.GetSpatialRef()
    outLayer = outDataSource.CreateLayer('', srs, ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(union_poly)

    outLayer.CreateFeature(outFeature)

    outFeature = None
    OutLayer = None
    OutDataSource = None


final_1D_boundary_dissolve = os.path.join(tempDir, 'final_1D_boundary_dissolve.shp')
dissolve_polygon(final_1D_boundary_singlepart, final_1D_boundary_dissolve)


print("Writing Projection file w/ hardcoded coordinate system.")
with open(os.path.join(tempDir,'test_boundary_polygon_erase_2DArea_dissolve.prj'), 'w') as f:
    f.write(coord_sys)
    f.close()

# clip final shapefiles to post-processing area shapefile

def clip_shapefile(inputDS, clipDS, outputDS, geom):

    ## Input
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    inDataSource = driver.Open(inputDS, 0)
    inLayer = inDataSource.GetLayer()

    print(inLayer.GetFeatureCount())
    ## Clip
    inClipSource = driver.Open(clipDS, 0)
    inClipLayer = inClipSource.GetLayer()
    print(inClipLayer.GetFeatureCount())

    ## Clipped Shapefile... Maybe???
    outDataSource = driver.CreateDataSource(outputDS)

    if geom == 'polygon':
        outLayer = outDataSource.CreateLayer('FINAL', geom_type=ogr.wkbMultiPolygon)
    if geom == 'polyline':
        outLayer = outDataSource.CreateLayer('FINAL', geom_type=ogr.wkbMultiLineString)

    print("Writing Projection file w/ hardcoded coordinate system.")
    outputDS_path = os.path.dirname(outputDS)
    outputDS_base = os.path.split(os.path.splitext(outputDS)[0])[1]
    with open(os.path.join(outputDS_path, outputDS_base +'.prj'), 'w') as f:
        f.write(coord_sys)
        f.close()

    ogr.Layer.Clip(inLayer, inClipLayer, outLayer)
    print(outLayer.GetFeatureCount())
    inDataSource.Destroy()
    inClipSource.Destroy()
    outDataSource.Destroy()

print("Now clipping final shapefiles to post-processing area polygon...")
print("...clipping 2D WSE Polygons...")
#Clip 2D WSE Polygons
poly_wse_shp_clipped = os.path.join(home_dir, 'test_polygon_all_data_clipped.shp')
clip_shapefile(poly_wse_shp, postp_area, poly_wse_shp_clipped, 'polygon')

print("...clipping 1D boundary area...")
#Clip 1D Boundary Polygon
final_1D_boundary_dissolve_clipped = os.path.join(home_dir, 'final_1D_boundary_dissolve_clipped.shp')
clip_shapefile(final_1D_boundary_dissolve, postp_area, final_1D_boundary_dissolve_clipped, 'polygon')

print("...clipping XS WSE polylines...")
#Clip XS WSE Results Polylines
test_XS_results_all_data = os.path.join(tempDir,'test_XS_results_all_data.shp')
test_XS_results_all_data_clipped = os.path.join(home_dir,'test_XS_results_all_data_clipped.shp')
clip_shapefile(test_XS_results_all_data, postp_area, test_XS_results_all_data_clipped, 'polyline')

