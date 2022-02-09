import sys
sys.path.append("/home/nilscp/GIT")

from pathlib import Path
from rastertools import utils

import geopandas as gpd
import pandas as pd
import rasterio as rio

# This should be replaced by a simple src.profile() from rasterio can get rid of
def readheader(ascii_DEM_filename):
    
    ''' 
    Read the header of a DEM .ascii file and extract the number of columns,
    rows, the origin of the DEM (lower left x- and y-coordinates), the cellsize 
    (i.e., pixel resolution) and Not-a-number value. 
    
    Parameters
    ----------
    ascii_DEM_filename : str
        absolute path to the ascii DEM file.
    
    Returns
    -------
    ncols : int
        Number of columns, width in pixels (along the x-direction) in the DEM.
    nrows : int
        Number of rows, height in pixels (along the y-direction) in the DEM.
    xllcorner : float
        Lower-left x coordinate. 
    yllcorner : float
        Lower-left y coordinate.
    cellsize : float
        Pixel resolution in units of proj. coordinate systems (most often in m)
    NODATA_value : float
        Not-a-number value.
    
    Example
    ----------    
    (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value) = readheader(filename)
    
    OK: 18.10.2018
    
    Changes
    ---------- 
    Modified the 12.06.2020: changed to absolute path
    '''
        
    lines =  []
    with open(ascii_DEM_filename) as f:
        ix = 0
        for line in f:
            if ix < 6:
                lines.append(line)
                ix = ix + 1
            else:
                break
    
    for ix, line in enumerate(lines):
        if ix == 0:
            tmp = line.strip('\n')
            ncols = int(tmp.split('ncols')[1])
        elif ix == 1:
            tmp = line.strip('\n')
            nrows = int(tmp.split('nrows')[1])
        elif ix == 2:
            tmp = line.strip('\n')
            xllcorner = float(tmp.split('xllcorner')[1])            
        elif ix == 3:
            tmp = line.strip('\n')
            yllcorner = float(tmp.split('yllcorner')[1])
        elif ix == 4:
            tmp = line.strip('\n')
            cellsize = float(tmp.split('cellsize')[1])
        else:
            tmp = line.strip('\n')
            NODATA_value = float(tmp.split('NODATA_value')[1])            
            
    return (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value)


# this will not work if the columns in the location of craters have different
# names.
def clip_raster_to_crater(location_of_craters, dem, clip_distance, output_dir, craterID = None):
        
    filename = Path(location_of_craters)
    
    # reading the shape file (craters)
    df = gpd.read_file(filename)
    
    # if a CRATER_ID is specified
    if craterID:
        df_selection = df[df.CRATER_ID == craterID]
    else:
        df_selection = df.copy()

    # loop through all craters or get result for a specific crater    
    for i in range(df_selection.shape[0]):
        
        # create a pandas DataFrame of a single crater entry
        dict_crater = {'geometry': [df_selection.geometry.iloc[i]], 'index': [0]}
        df_crater = pd.DataFrame(dict_crater)
        
        # create a geopandas Dataframe from the DataFrame
        geodataframe_crater = gpd.GeoDataFrame(df_crater, geometry=df_crater.geometry)
        geodataframe_crater.crs = df.crs
        
        # create a geopandas Series
        geom_geoseries = gpd.GeoSeries(geodataframe_crater.geometry)
        geom_geoseries.crs = df.crs
        
        # get the centroid from the crater ellipses
        centroids = geom_geoseries.centroid
    
        # Does it make the buffer from the polygon? I do specifically 
        buffer_array = centroids.buffer((df_selection.Diam_km.iloc[i]*0.5)*clip_distance*1000.0) 
        envelope_array = buffer_array.envelope
        
        tmp = envelope_array.__geo_interface__ 
        in_polygon = [tmp['features'][0]['geometry']]
        
        # generate name of the clipped raster
        clipped_raster = Path(output_dir) / (df_selection.CRATER_ID.iloc[i] + '.tif')
        utils.clip_advanced(dem, in_polygon, 'geojson', clipped_raster)
        
'''
location_of_craters = "/home/nilscp/GIT/crater_morphometry/data/rayed_craters/rayed_craters.shp"
dem = "/home/nilscp/tmp/Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd.tif"
craterID = "crater0096"
clip_distance = 8.0
output_dir = "/home/nilscp/tmp/"
       
clip_raster_to_crater(location_of_craters, 
                          dem, 
                          clip_distance, "/home/nilscp/tmp/", craterID)
'''
    