from rasterio.plot import reshape_as_raster, reshape_as_image
from pathlib import Path

import copy
import geopandas as gpd
import numpy as np
import pandas as pd
import random
import rasterio as rio
import scipy.linalg
import scipy.ndimage
import sys

sys.path.append("/home/nilscp/GIT/crater_morphometry")
import geomorphometry

def xy_circle(crater_radius, height_crater_center_px, width_crater_center_px):
    '''
    Parameters
    ----------
    crater_radius : float
        crater crater_radius in the unit of the proj. coordinate system (e.g., meters).
    height_crater_center_px : float
        centre of the crater (in pixel coordinates).
    width_crater_center_px : float
        centre of the crater (in pixel coordinates).

    Returns
    -------
    height_circle_coord : float
        x-coordinates (512 values equally spaced over the whole circle)
    width_circle_coord : float
        y-coordinates (512 values equally spaced over the whole circle)
    '''    
    # theta goes from 0 to 2pi
    theta = np.linspace(0.0, 2*np.pi, 512) #as in Geiger et al. (2013)
        
    # compute circle coordinates
    height_circle_coord = crater_radius*np.cos(theta) + height_crater_center_px
    width_circle_coord = crater_radius*np.sin(theta) + width_crater_center_px
    
    return (height_circle_coord, width_circle_coord)

def detrending(crater_radius,
               from_scaled_crater_radius, scaled_crater_radius_end,
               elevations, 
               dem_resolution, 
               height_mesh, width_mesh, 
               height_crater_center_px, width_crater_center_px, 
               filterMedianStd, debugging=False):
    
    '''
    Routine to fetch all the pixel values between SCALED_crater_radius_START and 
    SCALED_crater_radius_END from the crater centre (e.g., between 2.0 R and 3.0 R
    for detrending in a regional context and 0.9 and 1.1 for detrending
    assymetries along ) 

    Parameters
    ----------
    crater_radius : float
        crater crater_radius in the unit of the proj. coordinate system (e.g., meters).
    from_scaled_crater_radius : float
        Distance (scaled with the crater_radius of the crater diameter) from which the
        detrending step will be conducted. 
    scaled_crater_radius_end : float
        Distance (scaled with the crater_radius of the crater diameter) up to which the
        detrending step will be conducted. 
    elevations : numpy array
        Numpy array containg elevations (either original or detrended values).
    dem_resolution : float
        resolution of the DEM in meters.
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    height_crater_center_px : int
        centre of the crater in pixel coordinates.
    width_crater_center_px : int
        centre of the crater in pixel coordinates.
    filterMedianStd : boolean
        if True, values above the median of the elevation + one standard dev
        and below the median of the elevation - one standard dev are discarded.
        This allow 
    debugging : boolean
        if True returns coordinates where the detrending has been applied to

    Returns
    -------
    detrended_elevation : numpy array 
        Detrended elevations between the specified from_scaled_crater_radius and 
        .
        
    Suggestions for improvements
    -------
    - I am sure the selection could be made in a much more smooth way. Check
    https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle.
    This would be make this script much smaller and would avoid some unnecessary
    looping. 
    - Error message if, for some reasons, the from_scaled_crater_radius and scaled_crater_radius_end
    result in the fetching of elevations values outside of the mapped area.

    '''
    # in map coordinates       
    height_circle_px, width_circle_px = xy_circle((from_scaled_crater_radius*crater_radius) / dem_resolution, 
                       height_crater_center_px, width_crater_center_px)
    
    height_circle_px2, width_circle_px2 = xy_circle((scaled_crater_radius_end*crater_radius) / dem_resolution, 
                       height_crater_center_px, width_crater_center_px)
              
    (height_circle_px, width_circle_px, height_circle_px2, width_circle_px2) = (np.round(height_circle_px).astype('int'), np.round(width_circle_px).astype('int'),
                        np.round(height_circle_px2).astype('int'), np.round(width_circle_px2).astype('int'))

    # number of values along each cross section
    # sampled at two times the dem resolution
    n_points_along_cs = np.int32((crater_radius / dem_resolution) * 2.0)
    
    # empty array
    width_disk_selection_px = []
    height_disk_selection_px = []
    
    # Looping through all the x-coordinates of the circle located at 2R
    for i in range(len(height_circle_px)):

        # the starting coordinates move all the time and correspond to the 
        # boundary of the 2R circle
        height_cs_start = height_circle_px[i]
        width_cs_start = width_circle_px[i]
        
        # the end coordinates correspond to the boundary of the 3R circle
        height_cs_end = height_circle_px2[i]
        width_cs_end = width_circle_px2[i]
                
        # the distance is calculated, should be equal to two times the crater_radius
        (height_coord_cs, width_coord_cs) = (np.linspace(height_cs_start, height_cs_end, n_points_along_cs), 
                        np.linspace(width_cs_start, width_cs_end, n_points_along_cs))
        
        # only integer here
        (height_coord_cs, width_coord_cs) = (np.round(height_coord_cs).astype('int'), 
                                    np.round(width_coord_cs).astype('int'))
        
        #need to get rid of repetitions
        rep = np.zeros((len(height_coord_cs),2))
        rep[:,0] = height_coord_cs
        rep[:,1] = width_coord_cs
        __, index = np.unique(["{}{}".format(ix, j) for ix,j in rep], 
                              return_index=True)
        
        for i in index:
            height_disk_selection_px.append(height_coord_cs[i])
            width_disk_selection_px.append(width_coord_cs[i])
    
    
    # these correspond to all coordinates between the slice of 2R and 3R                
    height_disk_selection_px = np.array(height_disk_selection_px)
    width_disk_selection_px = np.array(width_disk_selection_px)
           
    # elevations are extracted for the map coordinates
    z = elevations[height_disk_selection_px,width_disk_selection_px]
 
    # and x- and y-coordinates in meters (this is correct now) # weird
    height_disk_selection = height_mesh[height_disk_selection_px,width_disk_selection_px]
    width_disk_selection = width_mesh[height_disk_selection_px,width_disk_selection_px]
          
    # the detrending routine is used (check that again)
    Z = elevation_plane(height_disk_selection, width_disk_selection, 
                           z, 
                           height_mesh, width_mesh, 
                           filterMedianStd)
    
    # the detrended linear plane is substracted to the data
    detrended_elevation = elevations - Z
    
    if debugging:
        return (height_disk_selection, width_disk_selection, z, detrended_elevation)
    else:
        return (detrended_elevation)

# changed name from linear3Ddetrending to elevation_plane
def elevation_plane(height_disk_selection, width_disk_selection, 
                    z, 
                    height_mesh, width_mesh, 
                    filterMedianStd):
    '''
    Based on the regional elevations

    Parameters
    ----------
    height_disk_selection : numpy array
        1-D numpy array containing all the x-coordinates of cells in between
        the from_scaled_crater_radius and scaled_crater_radius_end in the detrending function.
    width_disk_selection : numpy array
        1-D numpy array containing all the y-coordinates of cells in between
        the from_scaled_crater_radius and scaled_crater_radius_end in the detrending function.
    z : numpy array
        Elevations values within the disk of selected values. 
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    filterMedianStd : boolean
        if True, values above the median of the elevation + one standard dev
        and below the median of the elevation - one standard dev are discarded.

    Returns
    -------
    Z_plane : numpy array
        a linearly-fit elevation plane through the disk of specified elevation 
        values. The variabla has the same dimension as the original DEM or 
        detrended elevations.

    '''    
    if filterMedianStd:
    
        #median of the elevations
        zmed = np.nanmedian(z)
        zstd = np.nanstd(z)
        
        zidx = np.logical_and(z >= zmed - zstd, z <= zmed + zstd)
        
        # filter out the few elevation that stands out, which could change
        # the plane fitted through the points
        height_filtered = height_disk_selection[zidx]
        width_filtered = width_disk_selection[zidx]
        z_filtered = z[zidx]
        
        # Remove values that are more than one median +-std away
        cloud_points = np.c_[height_filtered,width_filtered,z_filtered]
        
    else:
        # we don't filter values out of the selection 
        cloud_points = np.c_[height_disk_selection,width_disk_selection,z]
       
    # best-fit linear plane (1st-order)
    A = np.c_[cloud_points[:,0], cloud_points[:,1], np.ones(cloud_points.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, cloud_points[:,2])    # coefficients
        
    # evaluate it on the original mesh grid
    Z_plane = C[0]*height_mesh + C[1]*width_mesh + C[2]
    
    return (Z_plane)

def local_maxima(array):
    '''
    Find local maximas across a 1-D numpy array.

    Parameters
    ----------
    array : 1-D numpy array

    Returns
    -------
    numpy mask array of local minimas (True/False)

    '''
    
    return ((array >= np.roll(array, 1)) &
            (array >= np.roll(array, -1)))

def maximum_elevation(z_profile, heights, widths, y_height_mesh,
                      x_width_mesh, y_height_crater_center_px,
                      x_width_crater_center_px):
    '''
    Find the maximum elevation along a cross section (1-D numpy array).

    Parameters
    ----------
    z : 1-D numpy array
        Elevations along a cross-section.
    heights : int
        all positions (in terms of indexes in the array) across the x-axis.
    widths : int
        all positions (in terms of indexes in the array) across the y-axis.
    y_height_mesh : 2D numpy array
        mesh grid with the same dimension as the elevations.
    x_width_mesh : 2D numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    y_height_coord_ME : numpy array
        position of the maximum_elevation in DEM map projection accross the x-axis.
    x_width_coord_ME : numpy array
        position of the maximum_elevation in DEM map projection accross the y-axis.
    y_height_px_ME : numpy array
        position of the maximum_elevation in px coordinates accross the x-axis.
    x_width_px_ME : numpy array
        position of the maximum_elevation in px coordinates accross the y-axis.
    maximum_elevation : float
        maximum elevation along the cross-section profile.

    '''
    
    # I still don't understand how nan values will occur, but just to be on the
    # safe side. The first and last values are just to avoid obvious
    # non-realistic
    max_elv_idx = np.argmax(z_profile)

    elev_ME = z_profile[max_elv_idx]

    y_height_px_ME = np.int32(np.round(heights[max_elv_idx]))
    x_width_px_ME = np.int32(np.round(widths[max_elv_idx]))

    y_height_coord_ME = y_height_mesh[y_height_px_ME, x_width_px_ME]
    x_width_coord_ME = x_width_mesh[y_height_px_ME, x_width_px_ME]

    # calculate the crater_radius to the global maximum elevation
    ab = (x_width_coord_ME - x_width_mesh[
        y_height_crater_center_px, x_width_crater_center_px]) ** 2.0
    bc = (y_height_coord_ME - y_height_mesh[
        y_height_crater_center_px, x_width_crater_center_px]) ** 2.0  # changed

    distance_to_ME_detection = np.sqrt(ab + bc)

    return (y_height_coord_ME, x_width_coord_ME,
            y_height_px_ME, x_width_px_ME,
            elev_ME, distance_to_ME_detection)




# changed name from local_elevation to local_elevations
# search_ncells is added so that not too many local maximas are detected
def local_elevations(z_profile, heights, widths, y_height_mesh,
                     x_width_mesh, y_height_crater_center_px,
                     x_width_crater_center_px, cs_id):
    '''
    Parameters
    ----------
    z : numpy 1-D array
        Elevations along the cross-section profile.
    n_points_along_cs : int
        Number of elevations along the cross section profile.
    search_ncells : int
        Number of cells to search before and after a local maxima is found.
    heights : numpy 1-D array (int)
        x-coordinate of all the elevations in the cross-section (in pixel).
    widths : numpy 1-D array (int)
        y-coordinate of all the elevations in the cross-section (in pixel).
    cs_id : int
        cross-section id.
    y_height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    x_width_mesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    y_height_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the x-axis).
    x_width_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the y-axis).
    y_height_px_LE : numpy 1-D array (int)
        positions of the local elevations in the DEM in pixels (across the x-axis).
    x_width_px_LE : numpy 1-D array (int)
        positions of the local elevations in the DEM in pixels (across the y-axis).
    elev_LE : numpy 1-D array (float)
        local elevations along the cross-section profile.
    profile_LE : int
        Several local elevations will be found per cross-section. We have thus
        the need to track the cross-section id.
    '''

    # need first to calculate the locations of the local maxima
    idx_local_minima = np.where(local_maxima(z_profile) == True)[0]
    idx_local_minima_filtered = idx_local_minima.astype('int')

    # should not be duplicates now that round is used
    y_height_px_LE = np.int32(np.round(heights[idx_local_minima_filtered]))
    x_width_px_LE = np.int32(np.round(widths[idx_local_minima_filtered]))

    ## you might have duplicate in there, especially because I am rounding

    y_height_coord_LE = y_height_mesh[y_height_px_LE, x_width_px_LE]
    x_width_coord_LE = x_width_mesh[y_height_px_LE, x_width_px_LE]

    # calculate the crater_radius to the global maximum elevation
    ab = (x_width_coord_LE - x_width_mesh[
        y_height_crater_center_px, x_width_crater_center_px]) ** 2.0
    bc = (y_height_coord_LE - y_height_mesh[
        y_height_crater_center_px, x_width_crater_center_px]) ** 2.0  # changed

    distance_to_LE_detection = np.sqrt(ab + bc)
    profile_LE = np.array(distance_to_LE_detection.size * [cs_id])
    elev_LE = z_profile[idx_local_minima_filtered]

    return (y_height_coord_LE, x_width_coord_LE,
            y_height_px_LE, x_width_px_LE,
            elev_LE, profile_LE, distance_to_LE_detection)

'''
Previous leftovers. I was using a searching distance to filter out 
some of the local elevations that were too far off. While this is good,
this decreases the quality of the automatic detection of the algorithm in 
the case where the initial centre of the crater is a bit off. Having a 
common projection, such as Equirectangular/Simple Cylindrical can cause
the centre of the crater to be  few hundreds of meters off. 

# upper-boundary
#ub = np.ceil(crater_radius + maximum_shift_LE)
# lower-boundary
#lb = np.floor(crater_radius - maximum_shift_LE)
#
#idx_selection = np.logical_and(distance_to_LE_detection <= ub,
#                               distance_to_LE_detection >= lb)


#distance_to_LE_detection = distance_to_LE_detection[idx_selection]
#nidx = idx_local_minima_filtered[idx_selection]
#elev_LE = z_profile[nidx]
#y_height_px_LE = np.int32(np.round(heights[nidx]))
#x_width_px_LE = np.int32(np.round(widths[nidx]))
#y_height_coord_LE = y_height_mesh[y_height_px_LE, x_width_px_LE]
#x_width_coord_LE = x_width_mesh[y_height_px_LE, x_width_px_LE]
#profile_LE = np.array(distance_to_LE_detection.size * [cs_id])
'''

def slope_change(crater_radius, z_profile, heights, widths, y_height_mesh,
                     x_width_mesh, y_height_crater_center_px,
                     x_width_crater_center_px, dem_resolution, cs_id):

    # calculate the crater_radius to the global maximum elevation
    ab = (widths - x_width_crater_center_px) ** 2.0
    bc = (heights - y_height_crater_center_px) ** 2.0  # changed
    distances = np.sqrt(ab + bc) * (dem_resolution)

    # create empty arrays    
    diff_slope = np.zeros(len(z_profile))

    # 0.05 R
    interval  = np.int32(np.ceil((crater_radius * 0.05) / (0.5*dem_resolution)))

    # need at least three points to calculate a slope with confidence
    # Otherwise, it does not work!
    if interval < 3:
        interval = 3

    # we need to test if these are the largest values within 0.1*R of the value
    for i in np.arange(interval,len(z_profile)-interval,1):

        slope_before = geomorphometry.cavity_slope(z_profile[i-interval:i],
                                                   distances[i-interval:i])
        slope_after = geomorphometry.cavity_slope(z_profile[i:i+interval],
                                                   distances[i:i+interval])

        diff_slope[i] = slope_after - slope_before

    # Let's take the 10 largest slope changes per cross section
    # Equivalent to finding the 10 smallest values
    #idx_selection = np.argpartition(diff_slope, 10)[:10]

    idx_selection = np.argmin(diff_slope)
    distance_to_CS_detection = distances[idx_selection]
    elev_CS = z_profile[idx_selection]
    y_height_px_CS = np.int32(np.round(heights[idx_selection]))
    x_width_px_CS = np.int32(np.round(widths[idx_selection]))
    y_height_coord_CS = y_height_mesh[y_height_px_CS, x_width_px_CS]
    x_width_coord_CS = x_width_mesh[y_height_px_CS, x_width_px_CS]
    profile_CS = np.array(distance_to_CS_detection.size * [cs_id])

    return (y_height_coord_CS, x_width_coord_CS,
            y_height_px_CS, x_width_px_CS,
            elev_CS, profile_CS, distance_to_CS_detection)

def find_nearest(array, value):
    '''
    Find nearest value in array and return both the value and index.

    Parameters
    ----------
    array : 1-D numpy array
    value : value to search for

    Returns
    -------
    array[idx] : float
        actual value in array (can be different from the searched value).
    idx : int
        index of where this value is found.

    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)

def detect_maximum_and_local_elevations(y_height_mesh, x_width_mesh,
                                        y_height_crater_center_px,
                                        x_width_crater_center_px,
                                        z_detrended, crater_radius,
                                        dem_resolution):
    '''
    Routine to extract the maximum elevations, local minima elevations and
    max. break in slope elevations from a given crater.

    Parameters
    ----------
    y_height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    x_width_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    y_height_crater_center_px : int
        centre of the crater in pixel coordinates.
    x_width_crater_center_px : int
        centre of the crater in pixel coordinates.
    elevations : numpy array
        Numpy array containg elevations (either original or detrended values).
    crater_radius : float
        crater crater_radius in the unit of the proj. coordinate system (e.g., meters).
    dem_resolution : float
        resolution of the DEM in meters.

    Returns
    -------
    None.

    '''
    # 2R circles pixel coordinates height_circle_px, width_circle_px
    height_circle_px, width_circle_px = xy_circle(
        (2.0 * crater_radius) / dem_resolution,
        y_height_crater_center_px, x_width_crater_center_px)

    # real values height_circle_px, width_circle_px
    (height_circle_px, width_circle_px) = (
    np.round(height_circle_px).astype('int'),
    np.round(width_circle_px).astype('int'))

    # we define the maximum elevation variables (need to be float for nan)
    y_height_coord_ME = np.ones(512) * np.nan
    x_width_coord_ME = np.ones(512) * np.nan
    y_height_px_ME = np.ones(512) * np.nan
    x_width_px_ME = np.ones(512) * np.nan
    elev_ME = np.ones(512) * np.nan
    radius_ME = np.ones(512) * np.nan
    profile_ME = np.arange(512)

    # we define the local maxima variables
    (y_height_coord_LE, x_width_coord_LE,
     y_height_px_LE, x_width_px_LE,
     elev_LE, profile_LE, radius_LE) = [np.array([]) for _ in range(7)]

    # we define the change in slope variables
    (y_height_coord_CS, x_width_coord_CS,
     y_height_px_CS, x_width_px_CS,
     elev_CS, profile_CS, radius_CS) = [np.array([]) for _ in range(7)]

    # samples at half the dem_resolution
    n_points_along_cs = np.int32(
        np.ceil(2.0 * crater_radius / dem_resolution) * 2.0)

    # generate cross sections between the centre of the crater and the 2.0R
    # circle pixel coordinates
    for ix in range(len(height_circle_px)):
        # find the pixel coordinates
        height_2R = height_circle_px[ix]
        width_2R = width_circle_px[ix]

        # the distance is calculated, should be equal to two times the crater_radius
        (heights, widths) = (
        np.linspace(y_height_crater_center_px, height_2R, n_points_along_cs),
        np.linspace(x_width_crater_center_px, width_2R, n_points_along_cs))

        # Extract the values along the line, using cubic interpolation and the
        # map coordinates
        z_profile = scipy.ndimage.map_coordinates(z_detrended,
                                          np.vstack((heights, widths)))

        # Detect the maximum elevation along each cross-section
        data_ME = maximum_elevation(
            z_profile, heights, widths, y_height_mesh,
            x_width_mesh, y_height_crater_center_px,
            x_width_crater_center_px)

        if data_ME:
            y_height_coord_ME[ix] = data_ME[0]
            x_width_coord_ME[ix] = data_ME[1]
            y_height_px_ME[ix] = data_ME[2]
            x_width_px_ME[ix] = data_ME[3]
            elev_ME[ix] = data_ME[4]
            radius_ME[ix] = data_ME[5]

        # Detect local elevations along each cross-section
        data_LE = local_elevations(
            z_profile, heights, widths, y_height_mesh,
            x_width_mesh, y_height_crater_center_px,
            x_width_crater_center_px, ix)

        # can be several local elevations per cross sections
        y_height_coord_LE = np.append(y_height_coord_LE, data_LE[0])
        x_width_coord_LE = np.append(x_width_coord_LE, data_LE[1])
        y_height_px_LE = np.append(y_height_px_LE, data_LE[2])
        x_width_px_LE = np.append(x_width_px_LE, data_LE[3])
        elev_LE = np.append(elev_LE, data_LE[4])
        profile_LE = np.append(profile_LE, data_LE[5])
        radius_LE = np.append(radius_LE, data_LE[6])

        # Detect change in slope along each cross-section

        data_CS = slope_change(crater_radius, z_profile, heights, widths,
                    y_height_mesh,
                    x_width_mesh, y_height_crater_center_px,
                    x_width_crater_center_px, dem_resolution, ix)

        # can be several local elevations per cross sections
        y_height_coord_CS = np.append(y_height_coord_CS, data_CS[0])
        x_width_coord_CS = np.append(x_width_coord_CS, data_CS[1])
        y_height_px_CS = np.append(y_height_px_CS, data_CS[2])
        x_width_px_CS = np.append(x_width_px_CS, data_CS[3])
        elev_CS = np.append(elev_CS, data_CS[4])
        profile_CS = np.append(profile_CS, data_CS[5])
        radius_CS = np.append(radius_CS, data_CS[6])


    return (y_height_coord_ME, x_width_coord_ME, y_height_px_ME, x_width_px_ME,
            elev_ME, profile_ME, radius_ME, y_height_coord_LE, x_width_coord_LE,
            y_height_px_LE, x_width_px_LE, elev_LE, profile_LE, radius_LE,
            y_height_coord_CS, x_width_coord_CS, y_height_px_CS,
            x_width_px_CS, elev_CS, profile_CS, radius_CS)


def rim_composite(crater_radius, y_height_coord_ME, x_width_coord_ME,
                  y_height_px_ME, x_width_px_ME, radius_ME,
                  y_height_coord_LE, x_width_coord_LE, y_height_px_LE,
                  x_width_px_LE, profile_LE, radius_LE,
                  y_height_coord_CS, x_width_coord_CS, y_height_px_CS,
                  x_width_px_CS, profile_CS, radius_CS,
                  maximum_shift_ME):
    # Let's hard code some of the values (we don't change them actually)
    starting_angle = [0, 45, 90, 135, 180, 225, 270, 315]

    # number of maximum elevation detections (always will be 512)
    # redundant cross-sections are filtered away in later step
    nME_detection = len(y_height_coord_ME)

    # equivalent points of 0, 45, 90, 135 degrees in terms of our data
    starting_crossS_id = (nME_detection * np.array(starting_angle)) / (360.)
    starting_crossS_id = starting_crossS_id.astype('int')
    n_iterations = len(starting_crossS_id) * 2  # ccw and cw

    y_height_coord_final = np.zeros((n_iterations, 512)) * np.nan
    x_width_coord_final = np.zeros((n_iterations, 512)) * np.nan
    y_height_px_final = np.zeros((n_iterations, 512)) * np.nan
    x_width_px_final = np.zeros((n_iterations, 512)) * np.nan
    profile_final = np.zeros((n_iterations, 512)) * np.nan
    flag_final = np.zeros((n_iterations, 512)) * np.nan

    # flag == 0 ---> Maximum elevation is used
    # flag == 1 ---> Change in slope is used
    # flag == 2 ---> Local elevation is used
    # flag == 4 ---> Gap

    # cumulative differences two consecutive cross sections (in terms of diameters)
    cum_delta_distance = np.zeros((n_iterations))
    gap = np.zeros((n_iterations))
    is_not_nME = np.zeros((n_iterations))

    '''
    ***********************LOOPS *********************************************
    '''
    pnum = 0

    for strt in starting_crossS_id:

        # counter clockwise loop through cross sections
        ccw = np.concatenate(
            (np.arange(strt, nME_detection), np.arange(0, strt)))

        # clockwise loop through cross sections
        cw = np.concatenate(((np.arange(strt + 1)[::-1]),
                             np.arange(strt + 1, nME_detection)[::-1]))

        # take both loops
        loops = [cw, ccw]

        # count the number of counter clockwise and clockwise loops
        # for example for a starting_angle = [0, 90, 180, 270], we should have 4
        # starting points looping clockwise and counterclockwise (2), so we will
        # have 8 candidates rim composite

        '''
        Example :
        starting_angle = 0
        ccw = [0...511]
        cw = [0, 511, 510 .... 1]
        '''

        for crossS_id in loops:
            k = 0

            # Find last point of loop to have reference for start
            # We need to find the maximum elevation for this profile
            distance_to_last_rim_composite_elevation = radius_ME[crossS_id[-1]]

            # this step can cause a problem, as if we start with a nan value
            # let's replace the nan value with the average crater radius if
            # this is the case
            #if np.isnan(distance_to_last_rim_composite_elevation):
            #    distance_to_last_rim_composite_elevation = crater_radius
            #else:
            #    None

            while k < 512:

                # loop through cross-sections (cs)
                i = crossS_id[k]

                # is crater_radius of ME at that cs within 0.10R of the previous estimate?
                ub = np.ceil(
                    distance_to_last_rim_composite_elevation + maximum_shift_ME)  # upper-boundary
                lb = np.floor(
                    distance_to_last_rim_composite_elevation - maximum_shift_ME)  # lower-boundary

                # Maximum elevation within 0.1 R is found
                if np.logical_and(radius_ME[i] <= ub,
                                  radius_ME[i] >= lb):

                    y_height_coord_final[pnum, k] = y_height_coord_ME[i]
                    x_width_coord_final[pnum, k] = x_width_coord_ME[i]
                    y_height_px_final[pnum, k] = y_height_px_ME[i]
                    x_width_px_final[pnum, k] = x_width_px_ME[i]
                    profile_final[pnum, k] = i
                    flag_final[pnum, k] = 0

                    cum_delta_distance[pnum] = cum_delta_distance[pnum] + np.abs(
                        distance_to_last_rim_composite_elevation - radius_ME[i])
                    distance_to_last_rim_composite_elevation = radius_ME[i]
                    k = k + 1

                else:

                    is_not_nME[pnum] = is_not_nME[pnum] + 1

                    # Change in slopes within 0.1 R
                    idx_CS_candidates = np.where(profile_CS == i)
                    CS_candidates = radius_CS[idx_CS_candidates]

                    #if np.logical_and(CS_candidates.size == 1, np.isnan(
                    # CS_candidates[0])):
                    #    any_CS_candidates = False
                    #else:
                    __, ilc_cs = find_nearest(CS_candidates,distance_to_last_rim_composite_elevation)
                    idx_CS = idx_CS_candidates[0][ilc_cs]
                    any_CS_candidates = np.logical_and(radius_CS[idx_CS] <= ub,radius_CS[idx_CS] >= lb)

                    # Local elevation within 0.1 R
                    idx_LE_candidates = np.where(profile_LE == i)
                    LE_candidates = radius_LE[idx_LE_candidates]

                    # Possible to have nan values in local elevations
                    #if np.logical_and(LE_candidates.size == 1, np.isnan(
                    # LE_candidates[0])):
                    #    any_LE_candidates = False
                    #else:
                    __, ilc_LE = find_nearest(LE_candidates,distance_to_last_rim_composite_elevation)
                    idx_LE = idx_LE_candidates[0][ilc_LE]
                    any_LE_candidates = np.logical_and(radius_LE[idx_LE] <= ub,radius_LE[idx_LE] >= lb)

                    # Local maxima within 0.1 R
                    if any_CS_candidates:

                        y_height_coord_final[pnum, k] = y_height_coord_CS[idx_CS]
                        x_width_coord_final[pnum, k] = x_width_coord_CS[idx_CS]
                        y_height_px_final[pnum, k] = y_height_px_CS[idx_CS]
                        x_width_px_final[pnum, k] = x_width_px_CS[idx_CS]
                        profile_final[pnum, k] = i
                        flag_final[pnum, k] = 1

                        cum_delta_distance[pnum] = cum_delta_distance[
                                                       pnum] + np.abs(
                            distance_to_last_rim_composite_elevation -
                            radius_CS[
                                idx_CS])
                        distance_to_last_rim_composite_elevation = radius_CS[
                            idx_CS]
                        k = k + 1

                    else:
                        if any_LE_candidates:
                            y_height_coord_final[pnum, k] = y_height_coord_LE[idx_LE]
                            x_width_coord_final[pnum, k] = x_width_coord_LE[idx_LE]
                            y_height_px_final[pnum, k] = y_height_px_LE[idx_LE]
                            x_width_px_final[pnum, k] = x_width_px_LE[idx_LE]
                            profile_final[pnum, k] = i
                            flag_final[pnum, k] = 2

                            cum_delta_distance[pnum] = cum_delta_distance[
                                                           pnum] + np.abs(
                                distance_to_last_rim_composite_elevation -
                                radius_LE[
                                    idx_LE])
                            distance_to_last_rim_composite_elevation = \
                            radius_LE[
                                idx_LE]
                            k = k + 1

                        else:
                            flag_final[pnum, k] = 4
                            gap[pnum] = gap[pnum] + 1
                            cum_delta_distance[pnum] = cum_delta_distance[
                                                           pnum] + np.abs(
                                distance_to_last_rim_composite_elevation)

                            if k == 511:
                                k = k + 1
                            else:
                                while k < 511:

                                    k = k + 1
                                    i = crossS_id[k]

                                    # Change in slopes within 0.1 R
                                    idx_CS_candidates = np.where(profile_CS == i)
                                    CS_candidates = radius_CS[idx_CS_candidates]

                                    #if np.logical_and(CS_candidates.size ==
                                    # 1,np.isnan(CS_candidates[0])):
                                    #    any_CS_candidates = False
                                    #else:
                                    __, ilc_CS = find_nearest(CS_candidates,
                                                           distance_to_last_rim_composite_elevation)
                                    idx_CS = idx_CS_candidates[0][ilc_CS]
                                    any_CS_candidates = np.logical_and(radius_CS[idx_CS] <= ub,radius_CS[idx_CS] >= lb)

                                    # Local elevation within 0.1 R (should always be at least one value)
                                    idx_LE_candidates = np.where(profile_LE == i)
                                    LE_candidates = radius_LE[idx_LE_candidates]

                                    # Possible to have nan values in local elevations
                                    #if np.logical_and(LE_candidates.size == 1,
                                    #                  np.isnan(LE_candidates[
                                    #                               0])):
                                    #    any_LE_candidates = False
                                    #else:
                                    __, ilc_LE = find_nearest(LE_candidates,
                                                              distance_to_last_rim_composite_elevation)
                                    idx_LE = idx_LE_candidates[0][ilc_LE]
                                    any_LE_candidates = np.logical_and(
                                        radius_LE[idx_LE] <= ub,
                                        radius_LE[idx_LE] >= lb)

                                    # Local slope within 0.1 R
                                    if any_CS_candidates:

                                        y_height_coord_final[pnum, k] = y_height_coord_CS[idx_CS]
                                        x_width_coord_final[pnum, k] = x_width_coord_CS[idx_CS]
                                        y_height_px_final[pnum, k] = y_height_px_CS[idx_CS]
                                        x_width_px_final[pnum, k] = x_width_px_CS[idx_CS]
                                        profile_final[pnum, k] = i
                                        flag_final[pnum, k] = 1

                                        cum_delta_distance[pnum] = cum_delta_distance[
                                                                       pnum] + np.abs(
                                            distance_to_last_rim_composite_elevation -
                                            radius_CS[idx_CS])
                                        distance_to_last_rim_composite_elevation = \
                                        radius_CS[idx_CS]
                                        is_not_nME[pnum] = is_not_nME[pnum] + 1
                                        k = k + 1
                                        break

                                    elif any_LE_candidates:

                                        y_height_coord_final[pnum, k] = y_height_coord_LE[
                                            idx_LE]
                                        x_width_coord_final[pnum, k] = x_width_coord_LE[
                                            idx_LE]
                                        y_height_px_final[pnum, k] = y_height_px_LE[idx_LE]
                                        x_width_px_final[pnum, k] = x_width_px_LE[idx_LE]
                                        profile_final[pnum, k] = i
                                        flag_final[pnum, k] = 2

                                        cum_delta_distance[pnum] = cum_delta_distance[
                                                                       pnum] + np.abs(
                                            distance_to_last_rim_composite_elevation -
                                            radius_LE[idx_LE])
                                        distance_to_last_rim_composite_elevation = \
                                        radius_LE[idx_LE]
                                        is_not_nME[pnum] = is_not_nME[pnum] + 1
                                        k = k + 1
                                        break

                                    else:
                                        is_not_nME[pnum] = is_not_nME[pnum] + 1
                                        flag_final[pnum, k] = 4
                                        gap[pnum] = gap[pnum] + 1
                                        cum_delta_distance[pnum] = \
                                        cum_delta_distance[
                                            pnum] + np.abs(
                                            distance_to_last_rim_composite_elevation)

            # to continue to loop through all possible clockwise, counterclockwise
            pnum += 1

    return (y_height_coord_final, x_width_coord_final,
            y_height_px_final, x_width_px_final,
            profile_final, flag_final, cum_delta_distance, is_not_nME, gap)
    
    
    
def first_run(crater_dem, crater_radius, scaling_factor):
    
    '''
    add a is_offset flag

    if offset is True,
    the centre of the crater is defined by the file specified in is_offset
    this will change the two lines 1287 and 1288

    The first run focus on re-adjusting the centre of the crater

    Parameters
    ----------
    crater_dem : TYPE
        DESCRIPTION.
    crater_radius : TYPE
        DESCRIPTION.
    scaling_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    dem_filename = Path(crater_dem)

    # --------------------------------------------------------------------------
    ###########################      LOADING DEM     ###########################
    with rio.open(crater_dem) as src:
        array = reshape_as_image(src.read())[:,:,0]
        meta = src.profile
    
    # infer dem resolution from the crater dem 
    dem_resolution = meta['transform'][0]
    
    # height scaling factor for the SLDEM2015
    z = array * scaling_factor

    # --------------------------------------------------------------------------
    ######################## FINDING CENTRE OF IMAGE ###########################

    y_height = np.linspace(0, (z.shape[0] - 1) * dem_resolution, z.shape[0])
    x_width = np.linspace(0, (z.shape[1] - 1) * dem_resolution, z.shape[1])

    # Crater centre in pixel coordinates
    y_height_crater_center_px = np.int32(z.shape[0] / 2)
    x_width_crater_center_px = np.int32(z.shape[1] / 2)

    # Origin of the raster (top left)
    x_width_origin = meta['transform'][2]
    y_height_origin = meta['transform'][5]

    # Crater centre in "world" projection
    y_height_center = y_height_origin - y_height[y_height_crater_center_px]
    x_width_center = x_width_origin + x_width[x_width_crater_center_px]

    y_height_mesh, x_width_mesh = np.meshgrid(y_height, x_width, indexing='ij')

    #---------------------------------------------------------------------------
    ###########################   DETRENDING 2R-3R   ###########################

    filterMedianStd = True  # use standard deviation removal

    # Only a single detrending in first run
    z_detrended = detrending(crater_radius,
                   2.0, 3.0,
                   z,
                   dem_resolution, 
                   y_height_mesh, x_width_mesh,
                   y_height_crater_center_px, x_width_crater_center_px,
                   filterMedianStd)

    #---------------------------------------------------------------------------
    ########################### SAVING DETRENDED DEM ###########################
    meta_detrended = meta.copy()
    meta_detrended['dtype'] = 'float64'

    dem_detrended = np.reshape(z_detrended,
                               (z_detrended.shape[0],
                                z_detrended.shape[1],
                                1))

    dem_detrended_folder = dem_filename.parent.with_name('dem_detrended')
    dem_detrended_folder.mkdir(parents=True, exist_ok=True)
    dem_detrended_filename = (dem_detrended_folder / (
            dem_filename.name.split('.tif')[0] + '_detrended.tif'))

    with rio.open(dem_detrended_filename, 'w', **meta_detrended) as ds:
        # reshape to rasterio raster format
        ds.write(reshape_as_raster(dem_detrended))

    #---------------------------------------------------------------------------
    ####### FINDING MAX ELEV., LOCAL MAX AND CHANGE IN ELEVATIONS ##############
    (y_height_coord_ME, x_width_coord_ME, y_height_px_ME, x_width_px_ME,
     elev_ME, profile_ME, radius_ME, y_height_coord_LE, x_width_coord_LE,
     y_height_px_LE, x_width_px_LE, elev_LE, profile_LE,
     radius_LE, y_height_coord_CS, x_width_coord_CS, y_height_px_CS,
            x_width_px_CS, elev_CS, profile_CS, radius_CS) = detect_maximum_and_local_elevations(y_height_mesh,
                                                      x_width_mesh,
                                                      y_height_crater_center_px,
                                                      x_width_crater_center_px,
                                                      z_detrended, crater_radius,
                                                      dem_resolution)

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE MAXIMUM ELEVATIONS ####################

    # update with more information
    df_xy_rim_comp = pd.DataFrame(
        {'id': list(np.arange(len(x_width_coord_ME))),
         'x': list(x_width_origin + x_width_coord_ME),
         'y': list(y_height_origin - y_height_coord_ME)})

    gdf_xy_rim_comp = gpd.GeoDataFrame(
        df_xy_rim_comp,
        geometry=gpd.points_from_xy(df_xy_rim_comp.x, df_xy_rim_comp.y,
                                    crs=str(meta['crs'])))

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)

    first_rim_comp = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_max_elevations.shp'))

    gdf_xy_rim_comp.to_file(first_rim_comp)

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE CHANGE IN SLOPE ######################

    df_xy_rim_comp = pd.DataFrame(
        {'id': list(np.arange(len(x_width_coord_CS))),
         'x': list(x_width_origin + x_width_coord_CS),
         'y': list(y_height_origin - y_height_coord_CS)})

    gdf_xy_rim_comp = gpd.GeoDataFrame(
        df_xy_rim_comp,
        geometry=gpd.points_from_xy(df_xy_rim_comp.x, df_xy_rim_comp.y,
                                    crs=str(meta['crs'])))

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)

    first_rim_comp = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_change_in_slopes.shp'))

    gdf_xy_rim_comp.to_file(first_rim_comp)

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE LOCAL MAXIMAS #########################

    df_xy_rim_comp = pd.DataFrame(
        {'id': list(np.arange(len(x_width_coord_LE))),
         'x': list(x_width_origin + x_width_coord_LE),
         'y': list(y_height_origin - y_height_coord_LE)})

    gdf_xy_rim_comp = gpd.GeoDataFrame(
        df_xy_rim_comp,
        geometry=gpd.points_from_xy(df_xy_rim_comp.x, df_xy_rim_comp.y,
                                    crs=str(meta['crs'])))

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)

    first_rim_comp = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_local_maximas.shp'))

    gdf_xy_rim_comp.to_file(first_rim_comp)



    #---------------------------------------------------------------------------
    ################ STITCHING OF THE FINAL RIM COMPOSITE ######################

    (y_height_coord_final, x_width_coord_final,
     y_height_px_final, x_width_px_final,
     profile_final, flag_final, cum_delta_distance, is_not_nME,
     gap) = rim_composite(crater_radius, y_height_coord_ME, x_width_coord_ME,
                          y_height_px_ME, x_width_px_ME, radius_ME,
                          y_height_coord_LE, x_width_coord_LE, y_height_px_LE,
                          x_width_px_LE, profile_LE, radius_LE, y_height_coord_CS,
                          x_width_coord_CS, y_height_px_CS, x_width_px_CS,
                          profile_CS, radius_CS, dem_resolution*2)

    #---------------------------------------------------------------------------
    ''' The stitching of the final rim composite returns 16 potential final 
    rim composite candidates. A final step consists at selecting the rim 
    which is the most complete by looking at the number of gaps and max 
    elevations in the rim composite.'''
    ################ STITCHING OF THE FINAL RIM COMPOSITE ######################

    # Ranking based on the three parameters
    data_dict = {'gap': gap, 'is_not_NME': is_not_nME, 'delta':
        cum_delta_distance}
    df = pd.DataFrame.from_dict(data_dict)
    df['ranking_gap'] = df['gap'].rank(na_option='bottom', pct=True)
    df['ranking_ME'] = df['is_not_NME'].rank(na_option='bottom', pct=True)
    df['ranking_delta'] = df['delta'].rank(na_option='bottom', pct=True)
    df['final_ranking'] = df['ranking_gap'] + df['ranking_ME'] + df['ranking_delta']
    df = df.sort_values(["final_ranking"], ascending = (True))
    best_rim_candidate = df.index[0]

    # can be nan values.... need to be fixed, number of nan values should be
    # equal to number of gaps
    idx = ~np.isnan(y_height_coord_final[best_rim_candidate])
    y_height_coord_rim = y_height_coord_final[best_rim_candidate][idx]
    x_width_coord_rim = x_width_coord_final[best_rim_candidate][idx]
    y_height_px_rim = np.int32(np.round(y_height_px_final[best_rim_candidate][idx]))
    x_width_px_rim = np.int32(np.round(x_width_px_final[best_rim_candidate][idx]))
    z_rim = z_detrended[y_height_px_rim, x_width_px_rim]
    profile_rim = profile_final[best_rim_candidate][idx]
    flag_rim = flag_final[best_rim_candidate][idx]

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE FINAL RIM COMPOSITE ###################

    df_xy_rim_comp = pd.DataFrame(
        {'id': list(profile_rim),
         'x': list(x_width_origin + x_width_coord_rim),
         'y': list(y_height_origin - y_height_coord_rim),
         'x_px': list(x_width_px_rim),
         'y_px': list(y_height_px_rim),
         'z_rim': list(z_rim),
         'type_of_elevation': list(flag_rim)})

    gdf_xy_rim_comp = gpd.GeoDataFrame(
        df_xy_rim_comp,
        geometry=gpd.points_from_xy(df_xy_rim_comp.x, df_xy_rim_comp.y,
                                    crs=str(meta['crs'])))

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)

    first_rim_comp = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_first_rim_comp.shp'))

    gdf_xy_rim_comp.to_file(first_rim_comp)

    #---------------------------------------------------------------------------
    ######## READJUSTING THE POSITION OF THE CENTRE OF THE CRATER ##############
    y_height_newcenter, x_width_newcenter, crater_radius_new, residu = \
        geomorphometry.leastsq_circle(
        y_height_coord_rim, x_width_coord_rim)

    #---------------------------------------------------------------------------
    ###### SAVING THE NEW & OLD POSITIONS OF THE CENTRE OF THE CRATER ##########

    # New position
    df_xy_new_center = pd.DataFrame(
        {'prof': [0], 'x_width': [x_width_newcenter + x_width_origin],
         'y_height': [y_height_origin - y_height_newcenter],
         'x_width_without_origin': [x_width_newcenter],
         'y_height_without_origin': [y_height_newcenter],
         'x_width_px': [np.int32(np.round(x_width_newcenter / dem_resolution))],
         'y_height_px': [np.int32(np.round(y_height_newcenter /
                                           dem_resolution))]})

    gdf_new_center = gpd.GeoDataFrame(
        df_xy_new_center.prof,
        geometry=gpd.points_from_xy(df_xy_new_center.x_width,
                                    df_xy_new_center.y_height,
                                    crs=str(meta_detrended['crs'])))


    new_crater_centre_filename = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_new_crater_center.shp'))

    gdf_new_center.to_file(new_crater_centre_filename)

    # Old position
    df_xy_old_center = pd.DataFrame(
        {'prof': [0],
         'x_width': [x_width_center],
         'y_height': [y_height_center],
         'x_width_without_origin': [x_width[x_width_crater_center_px]],
         'y_height_without_origin': [y_height[y_height_crater_center_px]],
         'x_width_px': [x_width_crater_center_px],
         'y_height_px': [y_height_crater_center_px]})

    gdf_old_center = gpd.GeoDataFrame(
        df_xy_old_center.prof,
        geometry=gpd.points_from_xy(df_xy_old_center.x_width,
                                    df_xy_old_center.y_height,
                                    crs=str(meta['crs'])))

    old_crater_centre_filename = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_old_crater_center.shp'))

    gdf_old_center.to_file(old_crater_centre_filename)


    return (y_height_newcenter, x_width_newcenter, y_height_mesh, x_width_mesh,
            crater_radius_new)

def second_run(crater_dem, scaling_factor, y_height_newcenter,
               x_width_newcenter, crater_radius_new):

    dem_filename = Path(crater_dem)

    # --------------------------------------------------------------------------
    ###########################      LOADING DEM     ###########################

    with rio.open(crater_dem) as src:
        array = reshape_as_image(src.read())[:, :, 0]
        meta = src.profile

    # infer dem resolution from the crater dem
    dem_resolution = meta['transform'][0]

    # height scaling factor for the SLDEM2015
    z = array * scaling_factor

    # Origin of the raster (top left)
    x_width_origin = meta['transform'][2]
    y_height_origin = meta['transform'][5]

    x_width_newcenter_px = np.int32(np.round(x_width_newcenter / dem_resolution))
    y_height_newcenter_px = np.int32(np.round(y_height_newcenter /
                                            dem_resolution))

    y_height = np.linspace(0, (z.shape[0] - 1) * dem_resolution, z.shape[0])

    x_width = np.linspace(0, (z.shape[1] - 1) * dem_resolution, z.shape[1])

    y_height_mesh, x_width_mesh = np.meshgrid(y_height, x_width, indexing='ij')

    #---------------------------------------------------------------------------
    ###########################   DETRENDING 2R-3R   ###########################

    filterMedianStd = True  # use standard deviation removal

    # Only a single detrending in first run
    z_detrended = detrending(crater_radius_new,
                   2.0, 3.0,
                   z,
                   dem_resolution,
                   y_height_mesh, x_width_mesh,
                   y_height_newcenter_px, x_width_newcenter_px,
                   filterMedianStd)

    #---------------------------------------------------------------------------
    ########################### SAVING DETRENDED DEM ###########################

    meta_detrended = meta.copy()
    meta_detrended['dtype'] = 'float64'

    dem_detrended = np.reshape(z_detrended,
                               (z_detrended.shape[0],
                                z_detrended.shape[1],
                                1))

    dem_detrended_folder = dem_filename.parent.with_name('dem_detrended')
    dem_detrended_folder.mkdir(parents=True, exist_ok=True)
    dem_detrended_filename = (dem_detrended_folder / (
            dem_filename.name.split('.tif')[0] + '_detrended.tif'))

    with rio.open(dem_detrended_filename, 'w', **meta_detrended) as ds:
        # reshape to rasterio raster format
        ds.write(reshape_as_raster(dem_detrended))
    # ---------------------------------------------------------------------------
    ########################   DETRENDING 0.9R-1.1R   ##########################

    filterMedianStd = False

    # Only a single detrending in first run
    z_detrended2 = detrending(crater_radius_new,
                             0.9, 1.1,
                             z_detrended,
                             dem_resolution,
                             y_height_mesh, x_width_mesh,
                             y_height_newcenter_px,
                             x_width_newcenter_px,
                             filterMedianStd)

    #---------------------------------------------------------------------------
    ########################### SAVING DETRENDED DEM ###########################
    meta_detrended2 = meta.copy()
    meta_detrended2['dtype'] = 'float64'

    dem_detrended2 = np.reshape(z_detrended2,
                               (z_detrended2.shape[0],
                                z_detrended2.shape[1],
                                1))

    dem_detrended_folder = dem_filename.parent.with_name('dem_detrended')
    dem_detrended_folder.mkdir(parents=True, exist_ok=True)
    dem_detrended_filename = (dem_detrended_folder / (
            dem_filename.name.split('.tif')[0] + '_detrended2.tif'))

    with rio.open(dem_detrended_filename, 'w', **meta_detrended2) as ds:
        # reshape to rasterio raster format
        ds.write(reshape_as_raster(dem_detrended2))

    #---------------------------------------------------------------------------
    ####### FINDING MAX ELEV., LOCAL MAX AND CHANGE IN ELEVATIONS ##############
    (y_height_coord_ME, x_width_coord_ME, y_height_px_ME, x_width_px_ME,
     elev_ME, profile_ME, y_height_coord_LE, x_width_coord_LE, y_height_px_LE,
     x_width_px_LE, elev_LE, profile_LE, y_height_coord_BS, x_width_coord_BS,
     y_height_px_BS, x_width_px_BS, elev_BS, prof_BS) = (
        detect_maximum_and_local_elevations(y_height_mesh,
                                            x_width_mesh,
                                            y_height_newcenter_px,
                                            x_width_newcenter_px,
                                            z_detrended,
                                            crater_radius_new,
                                            dem_resolution))

    # here it complains when I give the z_detrended2

    #---------------------------------------------------------------------------
    ################ STITCHING OF THE FINAL RIM COMPOSITE ######################

    # Maximum allowed radial discontinuity Drad
    # (I should convert these values in cells)
    maximum_shift_ME = 0.1 * crater_radius_new

    # Distance of interest (searching distance)
    maximum_shift_LE = 0.05 * crater_radius_new

    (candidates_rim_composite, n_ME_not_used, gaplist, delta_distances) = (
        rim_composite(y_height_coord_ME, x_width_coord_ME,
                      y_height_px_ME, x_width_px_ME,
                      elev_ME, profile_ME,
                      y_height_coord_LE, x_width_coord_LE,
                      y_height_px_LE, x_width_px_LE,
                      elev_LE, profile_LE,
                      y_height_mesh, x_width_mesh,
                      y_height_newcenter_px, x_width_newcenter_px,
                      maximum_shift_ME, maximum_shift_LE))

    #---------------------------------------------------------------------------
    ''' The stitching of the final rim composite returns 16 potential final 
    rim composite candidates. A final step consists at selecting the rim 
    which is the most complete by looking at the number of gaps and distances
    between of points of the rim composite.'''
    ################ STITCHING OF THE FINAL RIM COMPOSITE ######################
    for i in range(np.shape(candidates_rim_composite)[0]):
        a = np.mean(delta_distances[i]) + (0.5 * n_ME_not_used[i])
        if i == 0:
            b = a
            c = i
        else:
            if a < b:
                b = a
                c = i

    y_height_final = np.array(candidates_rim_composite[c][0, :])
    x_width_final = np.array(candidates_rim_composite[c][1, :])
    z_final = np.array(candidates_rim_composite[c][2, :])
    profile_final = np.array(candidates_rim_composite[c][3, :])
    flag_final = np.array(candidates_rim_composite[c][4, :])

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE FINAL RIM COMPOSITE ###################

    df_xy_rim_comp = pd.DataFrame(
        {'id': list(np.arange(len(x_width_final))),
         'x': list(x_width_origin + x_width_final),
         'y': list(y_height_origin - y_height_final)})

    gdf_xy_rim_comp = gpd.GeoDataFrame(
        df_xy_rim_comp.id,
        geometry=gpd.points_from_xy(df_xy_rim_comp.x, df_xy_rim_comp.y,
                                    crs=str(meta['crs'])))

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)

    second_rim_comp = (shp_folder / (
            dem_filename.name.split('.tif')[0] + '_second_rim_comp.shp'))

    gdf_xy_rim_comp.to_file(second_rim_comp)

    #---------------------------------------------------------------------------
    ############### CALCULATION OF GEOMORPHOMETRY PARAMETERS ###################

    data = geomorphometry.calculate(y_height, x_width,
                             y_height_final, x_width_final,
                             z_final, crater_radius_new,
                             dem_detrended2, dem_resolution,
                             y_height_newcenter_px, x_width_newcenter_px)

    return (data)


def geormorph_columns():

    columns = ['diameter_median', 'diameter_uncertainty', 'diameter_25p',
    'diameter_75p', 'diameter_max', 'depth_median', 'depth_uncertainty',
    'depth_25p', 'depth_75p', 'depth_max', 'rim_height_absolute_median',
    'rim_height_absolute_uncertainty',
    'rim_height_absolute_25p', 'rim_height_absolute_75p',
    'rim_height_absolute_max', 'rim_height_relative_median',
    'rim_height_relative_uncertainty',
    'rim_height_relative_25p', 'rim_height_relative_75p',
    'rim_height_relative_max', 'middle_cavity_slope_median',
    'middle_cavity_slope_uncertainty',
    'middle_cavity_slope_25p', 'middle_cavity_slope_75p',
    'middle_cavity_slope_max', 'upper_cavity_slope_median',
    'upper_cavity_slope_uncertainty',
    'upper_cavity_slope_25p', 'upper_cavity_slope_75p',
    'upper_cavity_slope_max', 'cavity_shape_exponent_median',
    'cavity_shape_exponent_uncertainty',
    'cavity_shape_exponent_25p', 'cavity_shape_exponent_75p',
    'cavity_shape_exponent_max', 'upper_cavity_roc_median',
    'upper_cavity_roc_uncertainty',
    'upper_cavity_roc_25p', 'upper_cavity_roc_75p', 'upper_cavity_roc_max',
    'upper_flank_roc_median', 'upper_flank_roc_uncertainty',
    'upper_flank_roc_25p', 'upper_flank_roc_75p', 'upper_flank_roc_max',
    'lower_rim_span_median', 'lower_rim_span_uncertainty',
    'lower_rim_span_25p', 'lower_rim_span_75p', 'lower_rim_span_max',
    'upper_rim_span_median', 'upper_rim_span_uncertainty',
    'upper_rim_span_25p', 'upper_rim_span_75p', 'upper_rim_span_max',
    'flank_slope_median', 'flank_slope_uncertainty',
    'flank_slope_25p', 'flank_slope_75p', 'flank_slope_max',
    'cavity_rim_decay_length_median', 'cavity_rim_decay_length_uncertainty',
    'cavity_rim_decay_length_25p', 'cavity_rim_decay_length_75p',
    'cavity_rim_decay_length_max',
    'flank_rim_decay_length_median', 'flank_rim_decay_length_uncertainty',
    'flank_rim_decay_length_25p', 'flank_rim_decay_length_75p',
    'flank_rim_decay_length_max']

    return (columns)

def main(location_of_craters, dem_folder, shp_folder,  scaling_factor,
         suffix, craterID = None):
    '''

    Returns
    -------
    '''
    #---------------------------------------------------------------------------
    ######################## Generate all variables  ###########################


    '''
    # so that I don't have to generate all variables.... 
    
    data = []
    for a, b, c in some_function_that_yields_data():
        data.append([a, b, c])

    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    '''

    dem_dummy = Path(dem_folder) / 'dummy.tif'
    shp_dummy = Path(shp_folder) / 'dummy.shp'
    filename = Path(location_of_craters)

    # reading the shape file (craters)
    gdf = gpd.read_file(filename)

    # if a CRATER_ID is specified
    if craterID:
        gdf_selection = gdf[gdf.CRATER_ID == craterID]
    else:
        gdf_selection = gdf.copy()

    for index, row in gdf_selection.iterrows():
        print(index)
        #if shp_dummy.with_name(row.CRATER_ID + suffix.split('.tif')[0] +
        #                       '_first_rim_comp.shp').is_file():
        #    None
        #else:
        first_run(dem_dummy.with_name(row.CRATER_ID + suffix),
                  row.diam/2.0, scaling_factor)

    def load_elevations(crater_dem, folder):

        crater_dem = Path(crater_dem)
        folder = (Path(folder) / 'dummy')
        crater_id = crater_dem.as_posix().split('/')[-1][:-4]

        df_me = gpd.read_file(folder.with_name(crater_id +
                                                   '_max_elevations.shp'))
        df_lm = gpd.read_file(folder.with_name(crater_id +
                                                   '_local_maximas.shp'))
        df_cs = gpd.read_file(folder.with_name(crater_id +
                                                   '_change_in_slopes.shp'))

        return (df_me, df_lm, df_cs)

'''
Example:
location_of_craters = '/home/nilscp/GIT/crater_morphometry/data/rayed_craters/rayed_craters_centroids.shp'
dem_folder = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/'
shp_folder = '/home/nilscp/tmp/fresh_impact_craters/shapefiles/'
scaling_factor = 0.5
suffix = '_LROKaguyaDEM.tif'
craterID = None
main(location_of_craters, dem_folder, shp_folder,  scaling_factor, suffix, craterID = None)

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0012_LROKaguyaDEM.tif'
crater_radius = 7830.5
scaling_factor = 0.5

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0000_LROKaguyaDEM.tif'
crater_radius = 11393.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius, scaling_factor)


crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0048_LROKaguyaDEM.tif'
crater_radius = 570.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius, scaling_factor)
'''



