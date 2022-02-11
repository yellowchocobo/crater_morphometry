import sys
sys.path.append("/home/nilscp/GIT")

from pathlib import Path
from rastertools import utils

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio

'''
This script will only work if the columns in the <location of craters> 
shapefile have the right names: diam (in m), lat and lon (in degrees). 

The data need to have the same coordinate system (e.g., Moon2000, 
# Mars2000). However, it does need to have the same projection. 

This will work:
- locations_of_craters --> lon, lat (Moon2000)
- A dem such as LRO_Kaguya_DEMmerge_60N60S.tif --> Equirectangular (Moon2000)

You can also change the radius of the coordinate system (if you want to 
switch between Mars2000 and Moon2000. 

'''
def Mars_2000():
    coord_sys = ('GEOGCS["Mars 2000",'
                 'DATUM["D_Mars_2000",'
                 'SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.89444722361179]],'
                 'PRIMEM["Greenwich",0],'
                 'UNIT["Decimal_Degree",0.0174532925199433]]')
    return(coord_sys)

def Moon_2000():
    coord_sys = ('GEOGCS["Moon 2000",'
                 'DATUM["D_Moon_2000",'
                 'SPHEROID["Moon_2000_IAU_IAG", 1737400.0, 0.0]],'
                 'PRIMEM["Greenwich", 0],'
                 'UNIT["Decimal_Degree", 0.0174532925199433]]')
    return(coord_sys)

def Moon_Equidistant_Cylindrical():
    proj = ('PROJCS["Moon_Equidistant_Cylindrical",'
            'GEOGCS["Moon 2000",DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Equidistant_Cylindrical"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",0],'
            'UNIT["Meter",1]]')

    return (proj)

def Moon_Mollweide(longitude):
    proj = ('PROJCS["Moon_Mollweide",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Mollweide"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))

    return (proj)

def Moon_Mercator():

    proj = ('PROJCS["Moon_Mercator_AUTO",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Mercator"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",0],'
            'UNIT["Meter",1]]')

    return (proj)

def Moon_Lambert_Conformal_Conic_N(longitude):
    proj = ('PROJCS["Moon_Lambert_Conformal_Conic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",30],'
            'PARAMETER["Standard_Parallel_2",60],'
            'PARAMETER["Latitude_Of_Origin",45],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))

    return(proj)


def Moon_Lambert_Conformal_Conic_S(longitude):
    proj = ('PROJCS["Moon_Lambert_Conformal_Conic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Lambert_Conformal_Conic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Standard_Parallel_1",-60],'
            'PARAMETER["Standard_Parallel_2",-30],'
            'PARAMETER["Latitude_Of_Origin",-45],'
            'UNIT["Meter",1]]')

    proj = proj.replace('_Meridian",0', '_Meridian",' + str(int(longitude)))

    return (proj)

def Moon_North_Pole_Stereographic():
    proj = ('PROJCS["Moon_North_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Stereographic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Scale_Factor",1],'
            'PARAMETER["Latitude_Of_Origin",90],'
            'UNIT["Meter",1]]')

def Moon_South_Pole_Stereographic():
    proj = ('PROJCS["Moon_South_Pole_Stereographic",'
            'GEOGCS["Moon 2000",'
            'DATUM["D_Moon_2000",'
            'SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["Decimal_Degree",0.0174532925199433]],'
            'PROJECTION["Stereographic"],'
            'PARAMETER["False_Easting",0],'
            'PARAMETER["False_Northing",0],'
            'PARAMETER["Central_Meridian",0],'
            'PARAMETER["Scale_Factor",1],'
            'PARAMETER["Latitude_Of_Origin",-90],'
            'UNIT["Meter",1]]')

def mollweide_proj(a, b):
    default_mol = ('+proj=moll +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400'
                   ' +units=m +no_defs')

    proj = default_mol.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return(proj)

def stereographic_npole(a,b):
    default_stereog = ('+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 '
                       '+a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_stereog.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return (proj)

def stereographic_spole(a,b):
    default_stereog = ('+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 '
                       '+a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_stereog.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return (proj)

def equirectangular_proj(longitude, latitude, a, b, default=True):

    if default:
        longitude = 0
        latitude = 0

    default_eqc = ('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0'
                   ' +a=1737400 +b=1737400 +units=m +no_defs')

    proj = default_eqc.replace('+lon_0=0','+lon_0=' + str(int(longitude)))
    proj = proj.replace('+lat_ts=0','+lat_ts=' + str(int(latitude)))
    proj = proj.replace('+a=1737400', '+a=' + str(int(a)))
    proj = proj.replace('+b=1737400', '+b=' + str(int(b)))

    return(proj)

def select_proj(longitude, latitude):

    if np.logical_and(latitude >= -30.0, latitude <= 30.0):
        proj = Moon_Equidistant_Cylindrical()
    elif np.logical_and(latitude < -30.0, latitude >= -60.0):
        proj = Moon_Lambert_Conformal_Conic_S(longitude)
    elif np.logical_and(latitude > 30.0, latitude <= 60.0):
        proj = Moon_Lambert_Conformal_Conic_N(longitude)
    elif latitude > 60.0:
        proj = Moon_North_Pole_Stereographic()
    elif latitude < -60.0:
        proj = Moon_South_Pole_Stereographic()

    return proj

def iterrows_calculations(gdf, dem, crs_dem, clip_distance, output_dir,
                          identifier):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for index, row in gdf.iterrows():
        crs_local = select_proj(row.lon,row.lat)
        crater_center = gpd.GeoSeries(row.geometry)
        crater_center.crs = gdf.crs
        crater_center_crs_local = crater_center.to_crs(crs_local)
        buff = crater_center_crs_local.buffer((row.diam / 2.0) * clip_distance)
        envelope = buff.envelope
        envelope_crs_dem = envelope.to_crs(crs_dem)
        poly = envelope_crs_dem.__geo_interface__
        in_poly = [poly['features'][0]['geometry']]


        # generate name of the clipped raster
        clipped_raster_fname = Path(output_dir) / (row.CRATER_ID + '_' +
                                                   identifier + '_eqc.tif')
        utils.clip_advanced(dem, in_poly, 'geojson', clipped_raster_fname)

        # reproject it to either equirectangular (no proj) or lambert
        crs_rasterio = rio.crs.CRS.from_wkt(crs_local)
        clipped_raster_fname_final = Path(output_dir) / (row.CRATER_ID + '_' +
                                                   identifier + '.tif')

        utils.reproject_raster(clipped_raster_fname, crs_rasterio,
                          clipped_raster_fname_final)




def clip_raster_to_crater(location_of_craters, dem, clip_distance,
                          output_dir, identifier, craterID=None):
        
    filename = Path(location_of_craters)
    
    # reading the shape file (craters)
    gdf = gpd.read_file(filename)

    # if a CRATER_ID is specified
    if craterID:
        gdf_selection = gdf[gdf.CRATER_ID == craterID]
    else:
        gdf_selection = gdf.copy()

    # extract the projection of the DEM
    with rio.open(dem) as src:
        meta = src.profile
    crs_dem = meta['crs'].to_wkt()

    iterrows_calculations(gdf_selection, dem, crs_dem, clip_distance,
                          output_dir, identifier)

'''
# Example:

location_of_craters = '/home/nilscp/GIT/crater_morphometry/data/rayed_craters/rayed_craters_centroids.shp'
dem = "/home/nilscp/QGIS/Moon/globalMosaics/Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd.tif"
orthoimage = "/home/nilscp/QGIS/Moon/globalMosaics/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
clip_distance = 8.0
output_dir = "/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/"
output_dir_ortho = "/home/nilscp/tmp/fresh_impact_craters/LROWAC_RayedCraters/"
identifier_dem = "LROKaguyaDEM"
identifier_orthoimage = "LROWAC"
craterID = 'crater0016'

# For a single crater
clip_raster_to_crater(location_of_craters, dem, clip_distance,
                          output_dir, identifier_dem, craterID = craterID)
                          
clip_raster_to_crater(location_of_craters, orthoimage, clip_distance,
                          output_dir_ortho, identifier_orthoimage, craterID = craterID)

# For all craters                          
clip_raster_to_crater(location_of_craters, dem, clip_distance,
                          output_dir, identifier_dem, craterID = None)
                          
clip_raster_to_crater(location_of_craters, orthoimage, clip_distance,
                          output_dir_ortho, identifier_orthoimage, craterID = None)

'''
    