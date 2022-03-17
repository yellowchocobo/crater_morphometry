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

from skimage.measure import CircleModel, EllipseModel
from shapely.geometry import Polygon
from tqdm import tqdm
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

sys.path.append("/home/nilscp/GIT/crater_morphometry")
import geomorphometry
from preprocessing import DEM_extraction

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

def maxima(array):
    return (array == np.max(array))

def minima(array):
    return (array == np.min(array))

def maximum_elevations(y_heights_px, x_widths_px, z_profiles, cross_sections_ids):

    idx_ME = np.apply_along_axis(maxima, 1, z_profiles)
    y_height_px_ME = y_heights_px[idx_ME]
    x_width_px_ME = x_widths_px[idx_ME]
    elev_ME = z_profiles[idx_ME]
    cross_sections_ME = cross_sections_ids[idx_ME]

    return (y_height_px_ME, x_width_px_ME, elev_ME, idx_ME, cross_sections_ME)


def local_elevations(y_heights_px, x_widths_px, z_profiles, cross_sections_ids):

    idx_LE = np.apply_along_axis(local_maxima, 1, z_profiles)
    y_height_px_LE = y_heights_px[idx_LE]
    x_width_px_LE = x_widths_px[idx_LE]
    elev_LE = z_profiles[idx_LE]
    cross_sections_LE = cross_sections_ids[idx_LE]


    return (y_height_px_LE, x_width_px_LE ,elev_LE, idx_LE, cross_sections_LE)

def maximum_slope_change(crater_radius, dem_resolution, y_heights_px,
                         x_widths_px, z_profiles, distances,
                         cross_sections_ids):

    interval = np.int32(np.ceil((crater_radius * 0.05) / (0.5 * dem_resolution)))

    # need at least three points to calculate a slope with confidence
    if interval < 3:
        interval = 3

    diff_slope = np.zeros_like(z_profiles)

    # This step takes a very long time, I am not sure if there is a better
    # way for doing that
    for i in range(z_profiles.shape[0]):
        for j in np.arange(interval, z_profiles.shape[1] - interval, 1):
            slope_before = geomorphometry.cavity_slope(
                z_profiles[i][j - interval:j],
                distances[i][j - interval:j])

            slope_after = geomorphometry.cavity_slope(
             z_profiles[i][j:j + interval],
             distances[i][j:j + interval])

            diff_slope[i,j] = slope_after - slope_before

    idx_CS = np.apply_along_axis(minima, 1, diff_slope)
    y_height_px_CS = y_heights_px[idx_CS]
    x_width_px_CS = x_widths_px[idx_CS]
    elev_CS = z_profiles[idx_CS]
    cross_sections_CS = cross_sections_ids[idx_CS]

    return (y_height_px_CS, x_width_px_CS ,elev_CS, idx_CS, cross_sections_CS)

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

def arange(array):
    return (array * np.arange(array.size))

def extract_cross_sections(crater_radius_px, dem_resolution,
                          y_height_crater_center_px,
                          x_width_crater_center_px,
                          z_detrended):

    # 2R circles pixel coordinates height_circle_px, width_circle_px
    height_circle_px, width_circle_px = xy_circle(
        2.0 * crater_radius_px,
        y_height_crater_center_px, x_width_crater_center_px)


    n_points_along_cs = np.int32(
        np.ceil(2.0 * crater_radius_px) * 2.0)

    # The circle is always divided in 512 camemberts :)
    z_profiles = np.ones((512,n_points_along_cs))
    y_heights_px = np.ones((512,n_points_along_cs))
    x_widths_px = np.ones((512,n_points_along_cs))
    cross_sections_ids = np.apply_along_axis(arange, 0, np.ones((512, n_points_along_cs)))

    # this step avoid a huge bottleneck in the script
    z_detrended = scipy.ndimage.spline_filter(z_detrended, order=3)

    # This step takes lot of time
    for ix in range(len(height_circle_px)):
        # find the pixel coordinates
        height_2R = height_circle_px[ix]
        width_2R = width_circle_px[ix]

        # the distance is calculated, should be equal to two times the crater_radius
        y_heights_px[ix] = np.linspace(y_height_crater_center_px, height_2R, n_points_along_cs)
        x_widths_px[ix] = np.linspace(x_width_crater_center_px, width_2R, n_points_along_cs)

        # Extract the values along the line, using cubic interpolation and the map coordinates
        # if prefilter = True --> Huge bottleneck!!!
        z_profiles[ix] = scipy.ndimage.map_coordinates(z_detrended, np.vstack((y_heights_px[ix], x_widths_px[ix])), order=3, prefilter=False)

    # calculate the crater_radius to the global maximum elevation
    h_d = (y_heights_px - y_height_crater_center_px)** 2.0
    w_d = (x_widths_px - x_width_crater_center_px)** 2.0
    distances = np.sqrt(h_d + w_d) * dem_resolution

    return y_heights_px, x_widths_px, z_profiles, distances, cross_sections_ids


def detect_potential_rim_candidates(y_heights_px, x_widths_px, z_profiles,
                                    crater_radius, cross_sections_ids,
                                    distances, dem_resolution):
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
    (y_height_px_ME, x_width_px_ME, elev_ME, idx_ME, cross_sections_ME) = (
    maximum_elevations(y_heights_px, x_widths_px, z_profiles,
                       cross_sections_ids))

    (y_height_px_LE, x_width_px_LE ,elev_LE, idx_LE, cross_sections_LE) = (
    local_elevations(y_heights_px, x_widths_px, z_profiles, cross_sections_ids))

    (y_height_px_CS, x_width_px_CS ,elev_CS, idx_CS, cross_sections_CS) = (
    maximum_slope_change(crater_radius, dem_resolution, y_heights_px,
                         x_widths_px, z_profiles, distances,
                         cross_sections_ids))

    return (y_height_px_ME, x_width_px_ME, elev_ME, idx_ME, cross_sections_ME,
            y_height_px_LE, x_width_px_LE, elev_LE, idx_LE, cross_sections_LE,
            y_height_px_CS, x_width_px_CS, elev_CS, idx_CS, cross_sections_CS)


def distance_between_points(y_height_loc_1, x_width_loc_1,
                            y_height_loc_2, x_width_loc_2):

    h_d = (y_height_loc_2 - y_height_loc_1)** 2.0
    w_d = (x_width_loc_2 - x_width_loc_1)** 2.0
    distances = np.sqrt(h_d + w_d)
    return distances

def find_candidates(y_height_px_CS, x_width_px_CS, cross_sections_CS,
                    y_height_px_ME, x_width_px_ME, cross_sections_ME,
                    y_height_px_LE, x_width_px_LE, cross_sections_LE,
                    y_height_last, x_width_last, i):

    y_height_candidate_CS = y_height_px_CS[cross_sections_CS == i]
    x_width_candidate_CS = x_width_px_CS[cross_sections_CS == i]

    y_height_candidate_ME = y_height_px_ME[cross_sections_ME == i]
    x_width_candidate_ME = x_width_px_ME[cross_sections_ME == i]

    y_height_candidate_LE = y_height_px_LE[cross_sections_LE == i]
    x_width_candidate_LE = x_width_px_LE[cross_sections_LE == i]

    dist_candidate_CS = distance_between_points(y_height_last,
                                                x_width_last,
                                                y_height_candidate_CS,
                                                x_width_candidate_CS)

    dist_candidate_ME = distance_between_points(y_height_last,
                                                x_width_last,
                                                y_height_candidate_ME,
                                                x_width_candidate_ME)

    dist_candidate_LE = distance_between_points(y_height_last,
                                                x_width_last,
                                                y_height_candidate_LE,
                                                x_width_candidate_LE)

    return (y_height_candidate_CS, x_width_candidate_CS, dist_candidate_CS,
            y_height_candidate_ME, x_width_candidate_ME, dist_candidate_ME,
            y_height_candidate_LE, x_width_candidate_LE, dist_candidate_LE)

def rim_composite(y_height_px_CS, x_width_px_CS,cross_sections_CS,
                  y_height_px_ME, x_width_px_ME,cross_sections_ME,
                  y_height_px_LE, x_width_px_LE,cross_sections_LE,
                  crater_radius, dem_resolution):

    # approx. distance in pixel between two points at a distance of the crater
    # radius
    px = np.tan(np.deg2rad(360./ 512.)) * (crater_radius / dem_resolution)
    px = np.int32(np.ceil(px))

    # Let's hard code some of the values (we don't change them actually)
    starting_angle = [0, 45, 90, 135, 180, 225, 270, 315]

    # number of maximum elevation detections (always will be 512)
    # redundant cross-sections are filtered away in later step
    nME_detection = len(y_height_px_ME) #y_height_px_ME.size

    # equivalent points of 0, 45, 90, 135 degrees in terms of our data
    starting_crossS_id = (nME_detection * np.array(starting_angle)) / (360.)
    starting_crossS_id = starting_crossS_id.astype('int')
    n_iterations = len(starting_crossS_id) * 2  # ccw and cw

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
    is_ME_or_CS = np.zeros((n_iterations))

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
            # Start from either a change in slope or a maximum elevation
            y_height_starting = y_height_px_CS[cross_sections_CS == crossS_id[-1]]
            x_width_starting = x_width_px_CS[cross_sections_CS == crossS_id[-1]]

            while k < 512:

                # loop through cross-sections (cs)
                i = crossS_id[k]

                # is the new candidate at the new cross-section within 5 pixels
                # of the previous position?
                (y_height_candidate_CS, x_width_candidate_CS, dist_candidate_CS,
                 y_height_candidate_ME, x_width_candidate_ME, dist_candidate_ME,
                 y_height_candidate_LE, x_width_candidate_LE,
                 dist_candidate_LE) = (find_candidates(
                    y_height_px_CS, x_width_px_CS, cross_sections_CS,
                    y_height_px_ME, x_width_px_ME, cross_sections_ME,
                    y_height_px_LE, x_width_px_LE,cross_sections_LE,
                    y_height_starting, x_width_starting, i))

                # is it within 5 pixels?
                if (dist_candidate_CS <= 5*px) or (dist_candidate_ME <= 5*px):

                    # who is the closest of max elevations and change in slope?
                    if np.min(dist_candidate_CS) < np.min(dist_candidate_ME):

                        y_height_px_final[pnum, k] = y_height_candidate_CS[0]
                        x_width_px_final[pnum, k] = x_width_candidate_CS[0]
                        profile_final[pnum, k] = i
                        flag_final[pnum, k] = 0

                        cum_delta_distance[pnum] = cum_delta_distance[pnum] + dist_candidate_CS[0]
                        is_ME_or_CS[pnum] = is_ME_or_CS[pnum] + 1
                        y_height_starting = y_height_candidate_CS[0]
                        x_width_starting = x_width_candidate_CS[0]
                        k = k + 1
                    else:
                        y_height_px_final[pnum, k] = y_height_candidate_ME[0]
                        x_width_px_final[pnum, k] = x_width_candidate_ME[0]
                        profile_final[pnum, k] = i
                        flag_final[pnum, k] = 1

                        cum_delta_distance[pnum] = cum_delta_distance[pnum] + np.abs(dist_candidate_ME[0])
                        is_ME_or_CS[pnum] = is_ME_or_CS[pnum] + 1
                        y_height_starting = y_height_candidate_ME[0]
                        x_width_starting = x_width_candidate_ME[0]
                        k = k + 1
                else:
                    if np.any(dist_candidate_LE <= 5*px):
                        idx = np.argmin(dist_candidate_LE)
                        y_height_px_final[pnum, k] = y_height_candidate_LE[idx]
                        x_width_px_final[pnum, k] = x_width_candidate_LE[idx]
                        profile_final[pnum, k] = i
                        flag_final[pnum, k] = 2
                        cum_delta_distance[pnum] = cum_delta_distance[pnum] + dist_candidate_LE[idx]
                        y_height_starting = y_height_candidate_LE[idx]
                        x_width_starting = x_width_candidate_LE[idx]
                        k = k + 1

                    else:
                        gap[pnum] = gap[pnum] + 1
                        flag_final[pnum, k] = 4

                        # add 5 pixels to cumulative difference
                        cum_delta_distance[pnum] = cum_delta_distance[pnum] + 5*px

                        if k == 511:
                            k = k + 1
                        else:
                            j = 0
                            while (k < 511) and (j < 5):
                                k = k + 1
                                i = crossS_id[k]

                                (y_height_candidate_CS, x_width_candidate_CS,
                                 dist_candidate_CS,
                                 y_height_candidate_ME, x_width_candidate_ME,
                                 dist_candidate_ME,
                                 y_height_candidate_LE, x_width_candidate_LE,
                                 dist_candidate_LE) = (find_candidates(
                                    y_height_px_CS, x_width_px_CS,
                                    cross_sections_CS,
                                    y_height_px_ME, x_width_px_ME,
                                    cross_sections_ME,
                                    y_height_px_LE, x_width_px_LE,
                                    cross_sections_LE,
                                    y_height_starting, x_width_starting, i))

                                if (dist_candidate_CS <= 5*px) or (dist_candidate_ME <= 5*px):

                                    # who is the closest of max elevations and change in slope?
                                    if np.min(dist_candidate_CS) < np.min(
                                            dist_candidate_ME):

                                        y_height_px_final[pnum, k] = \
                                        y_height_candidate_CS[0]
                                        x_width_px_final[pnum, k] = \
                                        x_width_candidate_CS[0]
                                        profile_final[pnum, k] = i
                                        flag_final[pnum, k] = 0

                                        cum_delta_distance[pnum] = cum_delta_distance[
                                                                       pnum] + \
                                                                   dist_candidate_CS[0]
                                        is_ME_or_CS[pnum] = is_ME_or_CS[
                                                                pnum] + 1
                                        y_height_starting = y_height_candidate_CS[0]
                                        x_width_starting = x_width_candidate_CS[0]
                                        k = k + 1
                                        break
                                    else:
                                        y_height_px_final[pnum, k] = \
                                        y_height_candidate_ME[0]
                                        x_width_px_final[pnum, k] = \
                                        x_width_candidate_ME[0]
                                        profile_final[pnum, k] = i
                                        flag_final[pnum, k] = 1

                                        cum_delta_distance[pnum] = cum_delta_distance[
                                                                       pnum] + np.abs(
                                            dist_candidate_ME[0])
                                        is_ME_or_CS[pnum] = is_ME_or_CS[
                                                                pnum] + 1
                                        y_height_starting = y_height_candidate_ME[0]
                                        x_width_starting = x_width_candidate_ME[0]
                                        k = k + 1
                                        break
                                else:
                                    if np.any(dist_candidate_LE <= 5*px):
                                        idx = np.argmin(dist_candidate_LE)

                                        y_height_px_final[pnum, k] = \
                                        y_height_candidate_LE[idx]
                                        x_width_px_final[pnum, k] = \
                                        x_width_candidate_LE[idx]
                                        profile_final[pnum, k] = i
                                        flag_final[pnum, k] = 2

                                        cum_delta_distance[pnum] = cum_delta_distance[
                                                                       pnum] + \
                                                                   dist_candidate_LE[
                                                                       idx]
                                        y_height_starting = y_height_candidate_LE[idx]
                                        x_width_starting = x_width_candidate_LE[idx]
                                        k = k + 1
                                        break
                                    else:
                                        j = j + 1
                                        gap[pnum] = gap[pnum] + 1
                                        # add 5 pixels to cumulative difference
                                        cum_delta_distance[pnum] = cum_delta_distance[pnum] + 5*px
                                        flag_final[pnum, k] = 4
                            while (j == 5) and (k < 511):
                                k = k + 1
                                gap[pnum] = gap[pnum] + 1
                                # add 5 pixels to cumulative difference
                                cum_delta_distance[pnum] = cum_delta_distance[
                                                               pnum] + 5 * px
                                flag_final[pnum, k] = 4

            pnum += 1

    # An ellipse is fitted through the 16 candidate portion of the crater rim
    (y_height_px_ellipse, x_width_px_ellipse, new_y_height_crater_center_px,
     new_x_width_crater_center_px, a, b, theta) = fit_ellipse(y_height_px_final,
                                                              x_width_px_final)

    return (y_height_px_ellipse, x_width_px_ellipse, new_y_height_crater_center_px, new_x_width_crater_center_px, a, b, theta)

def fit_ellipse(y_height_px_final, x_width_px_final):

    ell = EllipseModel()

    x_w_flatten = x_width_px_final.flatten()
    x_w_flatten = x_w_flatten[~np.isnan(x_w_flatten)]

    y_h_flatten =y_height_px_final.flatten()
    y_h_flatten = y_h_flatten[~np.isnan(y_h_flatten)]

    ell.estimate(np.column_stack((x_w_flatten,y_h_flatten)))

    x_width_crater_center_px, y_height_crater_center_px, a, b, theta = ell.params

    xy = EllipseModel().predict_xy(np.linspace(0.0, 2 * np.pi, 512),
                                   params=(x_width_crater_center_px, y_height_crater_center_px, a, b, theta))

    return (xy[:,1], xy[:,0], y_height_crater_center_px, x_width_crater_center_px, a, b, theta)

def cloud_points_to_shapefile(y_height_px, x_width_px, elev, cross_sections,
                              dem_resolution, y_height_origin, x_width_origin,
                              filename, meta):

    # filename
    filename = Path(filename)

    # convert from pixel to world coordinates
    x_width_coord = x_width_origin + x_width_px * dem_resolution
    y_height_coord = y_height_origin - y_height_px * dem_resolution

    df = pd.DataFrame(
        {'id': list(np.arange(x_width_px.size)),
         'x': list(x_width_coord),
         'y': list(y_height_coord),
         'x_px': list(x_width_px),
         'y_px': list(y_height_px),
         'elevation': list(elev),
         'cross_sect': list(cross_sections)})

    gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.x, df.y,
                                    crs=str(meta['crs'])))

    gdf.to_file(filename)

def generated_detrended_dem(location_of_craters, scaling_factor, dem_folder,
                            dem_detrended_folder, craterID=None, overwrite=False):

    dem_folder = Path(dem_folder)
    dems = list(sorted(dem_folder.glob("*.tif")))
    dem_detrended_dummy = Path(dem_detrended_folder) / 'dummy.tif'
    location_of_craters = Path(location_of_craters)

    # check if folder exists
    Path(dem_detrended_folder).mkdir(parents=True, exist_ok=True)

    # reading the shape file (craters)
    gdf = gpd.read_file(location_of_craters)

    # if a CRATER_ID is specified
    if craterID:
        gdf_selection = gdf[gdf.CRATER_ID == craterID]
    else:
        gdf_selection = gdf.copy()

    for index, row in tqdm(gdf_selection.iterrows(), total=gdf_selection.shape[0]):
        crater_dem = dems[index]
        crater_dem_detrended = dem_detrended_dummy.with_name(crater_dem.stem.split(".tif")[0] + "_detrended.tif")

        if not overwrite and crater_dem_detrended.is_file():
            None
        else:
            # ------------------------------------------------------------------
            ###########################      LOADING DEM     ###################
            with rio.open(crater_dem) as src:
                array = reshape_as_image(src.read())[:, :, 0]
                meta = src.profile

            # infer dem resolution from the crater dem
            dem_resolution = meta['transform'][0]

            # height scaling factor for the SLDEM2015
            z = array * scaling_factor

            # -----------------------------------------------------------------
            ######################## FINDING CENTRE OF IMAGE ##################

            # do we always want the centre of the crater to be at the middle? kind of...
            y_height = np.linspace(0, (z.shape[0] - 1) * dem_resolution, z.shape[0])
            x_width = np.linspace(0, (z.shape[1] - 1) * dem_resolution, z.shape[1])

            # Crater centre in pixel coordinates
            y_height_crater_center_px = np.int32(z.shape[0] / 2)
            x_width_crater_center_px = np.int32(z.shape[1] / 2)

            y_height_mesh, x_width_mesh = np.meshgrid(y_height, x_width, indexing='ij')

            # ---------------------------------------------------------------------------
            ###########################   DETRENDING 2R-3R   ###########################

            filterMedianStd = True  # use standard deviation removal

            # Only a single detrending in first run
            z_detrended = detrending(row.diam / 2.0,
                                     2.0, 3.0,
                                     z,
                                     dem_resolution,
                                     y_height_mesh, x_width_mesh,
                                     y_height_crater_center_px,
                                     x_width_crater_center_px,
                                     filterMedianStd)

            # ---------------------------------------------------------------------------
            ########################### SAVING DETRENDED DEM ###########################
            meta_detrended = meta.copy()
            meta_detrended['dtype'] = 'float64'

            dem_detrended = np.reshape(z_detrended,
                                       (z_detrended.shape[0],
                                        z_detrended.shape[1],
                                        1))

            with rio.open(crater_dem_detrended, 'w', **meta_detrended) as ds:
                # reshape to rasterio raster format
                ds.write(reshape_as_raster(dem_detrended))
    
def first_run(crater_dem, crater_radius, index):
    
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
        z_detrended = reshape_as_image(src.read())[:,:,0]
        meta = src.profile
    
    # infer dem resolution from the crater dem 
    dem_resolution = meta['transform'][0]

    # --------------------------------------------------------------------------
    ######################## FINDING CENTRE OF IMAGE ###########################

    # do we always want the centre of the crater to be at the middle? kind of...
    y_height = np.linspace(0, (z_detrended.shape[0] - 1) * dem_resolution, z_detrended.shape[0])
    x_width = np.linspace(0, (z_detrended.shape[1] - 1) * dem_resolution, z_detrended.shape[1])

    # Crater centre in pixel coordinates
    y_height_crater_center_px = np.int32(z_detrended.shape[0] / 2)
    x_width_crater_center_px = np.int32(z_detrended.shape[1] / 2)

    # Origin of the raster (top left)
    x_width_origin = meta['transform'][2]
    y_height_origin = meta['transform'][5]

    y_height_mesh, x_width_mesh = np.meshgrid(y_height, x_width, indexing='ij')

    #---------------------------------------------------------------------------
    ####### FINDING MAX ELEV., LOCAL MAX AND CHANGE IN ELEVATIONS ##############

    y_heights_px, x_widths_px, z_profiles, distances, cross_sections_ids = (
    extract_cross_sections(crater_radius/dem_resolution, dem_resolution,
                           y_height_crater_center_px,
                           x_width_crater_center_px,
                           z_detrended))

    (y_height_px_ME, x_width_px_ME, elev_ME, idx_ME, cross_sections_ME,
     y_height_px_LE, x_width_px_LE, elev_LE, idx_LE, cross_sections_LE,
     y_height_px_CS, x_width_px_CS, elev_CS, idx_CS,
     cross_sections_CS) = (detect_potential_rim_candidates(y_heights_px,
                                                          x_widths_px,
                                                          z_profiles,
                                                          crater_radius,
                                                          cross_sections_ids,
                                                          distances,
                                                          dem_resolution))

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE MAXIMUM ELEVATIONS ####################

    shp_folder = dem_filename.parent.with_name('shapefiles')
    shp_folder.mkdir(parents=True, exist_ok=True)
    filename = (shp_folder / (dem_filename.name.split('.tif')[0] + '_maximum_elevations.shp'))

    cloud_points_to_shapefile(y_height_px_ME, x_width_px_ME, elev_ME,
                              cross_sections_ME, dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)

    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE CHANGE IN SLOPE ######################

    filename= (shp_folder / (dem_filename.name.split('.tif')[0] + '_breaks_in_slope.shp'))

    cloud_points_to_shapefile(y_height_px_CS, x_width_px_CS, elev_CS,
                              cross_sections_CS, dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)
    #---------------------------------------------------------------------------
    ############### SAVE POSITION OF THE LOCAL MAXIMAS #########################

    filename= (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_local_maximas.shp'))

    cloud_points_to_shapefile(y_height_px_LE, x_width_px_LE, elev_LE,
                              cross_sections_LE, dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)

    #---------------------------------------------------------------------------
    ################ STITCHING OF THE FINAL RIM COMPOSITE ######################

    (y_height_px_ellipse, x_width_px_ellipse, new_y_height_crater_center_px,
     new_x_width_crater_center_px, a, b, theta) = (
        rim_composite(y_height_px_CS, x_width_px_CS,cross_sections_CS,
                      y_height_px_ME, x_width_px_ME,cross_sections_ME,
                      y_height_px_LE, x_width_px_LE,cross_sections_LE,
                      crater_radius, dem_resolution))

    x_width_px_ellipse_int = np.round(x_width_px_ellipse).astype('int')
    y_height_px_ellipse_int = np.round(y_height_px_ellipse).astype('int')
    elev_ellipse = z_detrended[y_height_px_ellipse_int,x_width_px_ellipse_int]

    filename= (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_ellipse_candidate_pts.shp'))

    cloud_points_to_shapefile(y_height_px_ellipse, x_width_px_ellipse, elev_ellipse,
                              np.arange(512), dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)

    #---------------------------------------------------------------------------
    ## SAVING THE NEW POSITIONS OF THE CENTRE OF THE CRATER + ELLIPSE INFO #####

    new_y_height_crater_center_px_int = np.round(new_y_height_crater_center_px).astype('int')
    new_x_width_crater_center_px_int = np.round(new_x_width_crater_center_px).astype('int')

    filename = (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_new_crater_centre.shp'))

    x_width_coord_ncenter = x_width_origin + x_width_mesh[new_y_height_crater_center_px_int, new_x_width_crater_center_px_int]
    y_height_coord_ncenter = y_height_origin - y_height_mesh[new_y_height_crater_center_px_int, new_x_width_crater_center_px_int]

    df = pd.DataFrame(
        {'id': [index],
         'x': [x_width_coord_ncenter],
         'y': [y_height_coord_ncenter],
         'xc_width_px': [new_x_width_crater_center_px],
         'yc_height_px': [new_y_height_crater_center_px],
         'a': [a],
         'b': [b],
         'theta': [theta]})

    gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.x, df.y,
                                    crs=str(meta['crs'])))

    gdf.to_file(filename)

    #---------------------------------------------------------------------------
    ########################### REFINING OF THE ELLIPSE ########################

    ell = EllipseModel()
    ell2 = EllipseModel()
    ell.params = new_x_width_crater_center_px, new_y_height_crater_center_px, 0.9*a, 0.9*b, theta
    ell2.params = new_x_width_crater_center_px, new_y_height_crater_center_px, 1.2*a, 1.2*b, theta

    xy = ell.predict_xy(np.linspace(0.0, 2 * np.pi, 512))
    xy2 = ell2.predict_xy(np.linspace(0.0, 2 * np.pi, 512))

    p1 = Polygon(xy)
    p2 = Polygon(xy2)

    gdf_ME = is_within_ellipse(p1, p2, y_height_px_ME, x_width_px_ME, elev_ME, cross_sections_ME)
    gdf_CS = is_within_ellipse(p1, p2, y_height_px_CS, x_width_px_CS, elev_CS, cross_sections_CS)

    # refined ellipse
    y_height_px_refined = np.concatenate((gdf_ME.geometry.y.values,
                                          gdf_CS.geometry.y.values))

    x_width_px_refined = np.concatenate((gdf_ME.geometry.x.values,
                                         gdf_CS.geometry.x.values))

    elev_refined = np.concatenate((gdf_ME.elevation.values,
                                         gdf_CS.elevation.values))

    cross_sections_refined = np.concatenate((gdf_ME.cross_sect.values,
                                         gdf_CS.cross_sect.values))


    filename= (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_points_used_for_ellipse_candidate2.shp'))

    # the points used for the new ellipse.
    cloud_points_to_shapefile(y_height_px_refined, x_width_px_refined, elev_refined,
                              cross_sections_refined, dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)

    new_ell = EllipseModel()
    new_ell.estimate(np.column_stack((x_width_px_refined, y_height_px_refined)))
    xc_ref_ell, yc_ref_ell, a_ref_ell, b_ref_ell, theta_ref_ell = new_ell.params
    new_ell_pred = new_ell.predict_xy(np.linspace(0.0, 2 * np.pi, 512))

    y_ell_pred_px_int = np.round(new_ell_pred[:,1]).astype('int')
    x_ell_pred_px_int = np.round(new_ell_pred[:,0]).astype('int')
    elev_refined_ellipse = z_detrended[y_ell_pred_px_int,
                                       x_ell_pred_px_int]

    # ellipse

    filename= (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_ellipse_candidate2_pts.shp'))

    cloud_points_to_shapefile(new_ell_pred[:,1], new_ell_pred[:,0], elev_refined_ellipse,
                              np.arange(512), dem_resolution,
                              y_height_origin, x_width_origin,
                              filename, meta)

    #---------------------------------------------------------------------------
    ### SAVING AGAIN THE NEW POSITIONS OF THE CENTRE OF THE CRATER + ELLIPSE ###

    y_height_px_refined_ellipse_centre_px_int = np.round(yc_ref_ell).astype('int')
    x_width_px_refined_ellipse_centre_px_int = np.round(xc_ref_ell).astype('int')

    filename = (shp_folder / (dem_filename.name.split('.tif')[0] +
                             '_new_crater_centre2.shp'))

    x_width_coord_ncenter = x_width_origin + x_width_mesh[y_height_px_refined_ellipse_centre_px_int, x_width_px_refined_ellipse_centre_px_int]
    y_height_coord_ncenter = y_height_origin - y_height_mesh[y_height_px_refined_ellipse_centre_px_int, x_width_px_refined_ellipse_centre_px_int]

    # calculate new diameter
    new_diam = (((a*dem_resolution) + (b*dem_resolution)) / 2.0) * 2.0

    df = pd.DataFrame(
        {'id': [index],
         'x': [x_width_coord_ncenter],
         'y': [y_height_coord_ncenter],
         'xc_px': [xc_ref_ell],
         'yc_px': [yc_ref_ell],
         'a': [a_ref_ell],
         'b': [b_ref_ell],
         'theta': [theta_ref_ell],
         'diam': [new_diam],
         'res' : [dem_resolution]})

    gdf = gpd.GeoDataFrame(df,
        geometry=gpd.points_from_xy(df.x, df.y,
                                    crs=str(meta['crs'])))

    gdf.to_file(filename)

    #---------------------------------------------------------------------------
    ### SAVING A POLYGON SHAPEFILE ELLIPSE with INFO ON ELLIPSE  ###
    ### Both in world (for visualization) and pixel coordinates ###

    # do I need to use int actually?
    x_width_ellipse_coord = x_width_origin + new_ell_pred[:, 0] * dem_resolution
    y_height_ellipse_coord = y_height_origin - new_ell_pred[:, 1] * dem_resolution

    x_width_ellipse_coord_px = x_width_origin + new_ell_pred[:, 0] * dem_resolution
    y_height_ellipse_coord_px = y_height_origin - new_ell_pred[:, 1] * dem_resolution

    poly = Polygon(list(zip(x_width_ellipse_coord,y_height_ellipse_coord)))
    filename = (shp_folder / (dem_filename.name.split('.tif')[0] + '_ellipse_candidate2_polygon.shp'))
    gdf.geometry = [poly]
    gdf.to_file(filename)

    poly = Polygon(list(zip(new_ell_pred[:, 0],new_ell_pred[:, 1])))
    filename = (shp_folder / (dem_filename.name.split('.tif')[0] + '_ellipse_candidate2_polygon_px.shp'))
    gdf.geometry = [poly]
    gdf.to_file(filename)


def is_within_ellipse(p1, p2, y_height, x_width, elev, cross_section):

    gdf = gpd.GeoDataFrame({'cross_sect': cross_section, 'elevation': elev},
        geometry=gpd.points_from_xy(x_width, y_height))

    index_selection = []
    for index, row in gdf.iterrows():
        if not row.geometry.within(p1):
            if row.geometry.within(p2):
                index_selection.append(index)

    return (gdf.iloc[index_selection])


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

def main(location_of_craters, dem_folder, shp_folder,
         suffix, threshold=None, craterID=None):
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

    for index, row in tqdm(gdf_selection.iterrows(), total=gdf_selection.shape[0]):
        if shp_dummy.with_name(row.CRATER_ID + suffix.split('.tif')[0] +
                               '_ellipse_candidate2.shp').is_file():
            None
        else:
            try:
                if threshold and row.diam < threshold:
                    first_run(dem_dummy.with_name(row.CRATER_ID + suffix),
                              row.diam/2.0, index)
                elif threshold and row.diam > threshold:
                    None
                else:
                    first_run(dem_dummy.with_name(row.CRATER_ID + suffix),
                              row.diam/2.0, index)
            except:
                print("some problem here")

def load_ellipse_candidate(crater_dem, dem_detrended, folder):

    crater_dem = Path(crater_dem)
    dem_detrended = Path(dem_detrended)
    crater_id = crater_dem.as_posix().split('/')[-1][:-4]
    folder = Path(folder) / 'dummy'

    # --------------------------------------------------------------------------
    ###########################      LOADING DEM     ###########################
    with rio.open(dem_detrended) as src:
        array = reshape_as_image(src.read())[:, :, 0]
        meta = src.profile

    # infer dem resolution from the crater dem
    dem_resolution = meta['transform'][0]

    # --------------------------------------------------------------------------
    ######################## FINDING CENTRE OF IMAGE ###########################

    y_height = np.linspace(0, (array.shape[0] - 1) * dem_resolution, array.shape[0])
    x_width = np.linspace(0, (array.shape[1] - 1) * dem_resolution, array.shape[1])

    # Origin of the raster (top left)
    x_width_origin = meta['transform'][2]
    y_height_origin = meta['transform'][5]

    y_height_mesh, x_width_mesh = np.meshgrid(y_height_origin - y_height, x_width_origin + x_width, indexing='ij')

    gdf_me = gpd.read_file(folder.with_name(crater_id + '_max_elevations.shp'))
    gdf_lm = gpd.read_file(folder.with_name(crater_id + '_local_maximas.shp'))
    gdf_cs = gpd.read_file(folder.with_name(crater_id + '_change_in_slopes.shp'))
    gdf_ell = gpd.read_file(folder.with_name(crater_id +'_ellipse_candidate2.shp'))
    gdf_points_ell = gpd.read_file(folder.with_name(crater_id + '_points_used_for_ellipse_candidate2.shp'))
    gdf_centre = gpd.read_file(folder.with_name(crater_id +'_new_crater_centre.shp'))
    gdf_centre2 = gpd.read_file(folder.with_name(crater_id + '_new_crater_centre2.shp'))

    ell = EllipseModel()
    cic = CircleModel()

    ell.params = (gdf_centre2.xc_width_p.values, gdf_centre2.yc_height_.values,
                  gdf_centre2.a.values, gdf_centre2.b.values,
                  gdf_centre2.theta.values)

    crater_radius_px_from_ellipse = (gdf_centre2.a.values + gdf_centre2.b.values) / 2.0

    cic.params = (gdf_centre2.xc_width_p.values, gdf_centre2.yc_height_.values,
                  crater_radius_px_from_ellipse*2.0)

    xy_ell = ell.predict_xy(np.linspace(0.0, 2 * np.pi, 512))
    xy_cic = cic.predict_xy(np.linspace(0.0, 2 * np.pi, 512))

    plt.plot(xy_cic[:, 0], xy_cic[:, 1], "g", lw=2)
    plt.plot(xy_ell[:, 0], xy_ell[:, 1], "b", lw=2)

    y_heights_px, x_widths_px, z_profiles, distances, cross_sections_ids = extract_cross_sections(crater_radius_px_from_ellipse[0],
                                                                                                  dem_resolution,
                                                                                                  gdf_centre2.yc_height_.values[0],
                                                                                                  gdf_centre2.xc_width_p.values[0], array)

    # finding where the ellipse is
    dd = distance_between_points(gdf_centre2.yc_height_.values[0],
                                 gdf_centre2.xc_width_p.values[0],
                                 gdf_ell.y_px.iloc[0], gdf_ell.x_px.iloc[0])

    dist_dd = dd  * dem_resolution


    test = np.column_stack((xy_ell[:,1], xy_ell[:,0]))
    __, unique_index = np.unique(["{}{}".format(i, j) for i,j in test],
                          return_index=True)

    unique_i = test[unique_index,:]
    prof_uni_detected = cs[unique_index] #still contain zeros


    return (gdf_me, gdf_lm, gdf_cs, gdf_ell, gdf_centre)


def update_global_crater_centres(in_crater_centres_shp, shapefiles_folder, out_crater_centres_shp):
    in_crater_centres_shp = Path(in_crater_centres_shp)
    shapefiles_dummy = Path(shapefiles_folder) / "dummy.shp"
    gdf = gpd.read_file(in_crater_centres_shp)
    crs = gdf.crs.to_wkt()

    data = []
    for index, row in gdf.iterrows():
        try:
            shp = shapefiles_dummy.with_name(row.CRATER_ID + "_LROKaguyaDEM_new_crater_centre2.shp")
            gdf_tmp = gpd.read_file(shp)
            gdf_ncrs = gdf_tmp.to_crs(crs)
            tmp_list = list(gdf_ncrs.values[0]) + [gdf_tmp.crs.to_wkt()]
            tmp_list[0] = row.CRATER_ID
            data.append(tmp_list)
        except:
            None

    df = pd.DataFrame(data, columns=['CRATER_ID', 'x', 'y',
                                     'xc_width', 'yc_height',
                                     'a', 'b', 'theta', 'geometry', 'crs'])

    gdf_ncenters = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf_ncenters.to_file(out_crater_centres_shp)



def update_global_ellipse(in_crater_centres_shp, shapefiles_folder):
    in_crater_centres_shp = Path(in_crater_centres_shp)
    shapefiles_dummy = Path(shapefiles_folder) / "dummy.shp"
    eqc_name = "/home/nilscp/tmp/fresh_impact_craters/shapefiles/global/global_ellipse_candidate2_eqc.shp"
    conical_south = "/home/nilscp/tmp/fresh_impact_craters/shapefiles/global/global_ellipse_candidate2_conicN.shp"
    conical_north = "/home/nilscp/tmp/fresh_impact_craters/shapefiles/global/global_ellipse_candidate2_conicS.shp"

    crs_north = DEM_extraction.Moon_Lambert_Conformal_Conic_N(45.0)
    crs_south = DEM_extraction.Moon_Lambert_Conformal_Conic_S(-45.0)
    crs_eqc = DEM_extraction.Moon_Equidistant_Cylindrical()

    gdf = gpd.read_file(in_crater_centres_shp)

    data_eqc = []
    data_conicalN = []
    data_conicalS = []

    for index, row in gdf.iterrows():
        try:
            shp = shapefiles_dummy.with_name(row.CRATER_ID + "_LROKaguyaDEM_ellipse_candidate2.shp")
            gdf_tmp = gpd.read_file(shp)
            if np.logical_and(row.lat >= -30.0, row.lat <= 30.0):
                gdf_ncrs = gdf_tmp.to_crs(crs_eqc)
                tmp_list = list(zip(len(gdf_ncrs.geometry.values) * [row.CRATER_ID], gdf_ncrs.geometry.values))
                tmp_list[0] = row.CRATER_ID
                data_eqc.append(tmp_list)

            elif np.logical_and(row.lat < -30.0, row.lat >= -60.0):
                gdf_ncrs = gdf_tmp.to_crs(crs_south)
                tmp_list = list(zip(len(gdf_ncrs.geometry.values) * [row.CRATER_ID], gdf_ncrs.geometry.values))
                tmp_list[0] = row.CRATER_ID
                data_conicalS.append(tmp_list)

            elif np.logical_and(row.lat > 30.0, row.lat <= 60.0):
                gdf_ncrs = gdf_tmp.to_crs(crs_north)
                tmp_list = list(zip(len(gdf_ncrs.geometry.values) * [row.CRATER_ID], gdf_ncrs.geometry.values))
                tmp_list[0] = row.CRATER_ID
                data_conicalN.append(tmp_list)
        except:
            None

    df_eqc = pd.DataFrame(data_eqc, columns=['CRATER_ID', 'geometry'])

    df_north = pd.DataFrame(data_conicalN, columns=['CRATER_ID', 'geometry'])

    df_south = pd.DataFrame(data_conicalS, columns=['CRATER_ID', 'geometry', 'crs'])


    gdf_eqc = gpd.GeoDataFrame(df_eqc, geometry=df_eqc.geometry, crs=crs_eqc)
    gdf_eqc.to_file(eqc_name)

    gdf_north = gpd.GeoDataFrame(df_north, geometry=df_north.geometry, crs=crs_north)
    gdf_north.to_file(conical_north)

    gdf_south = gpd.GeoDataFrame(df_south, geometry=df_south.geometry, crs=crs_south)
    gdf_south.to_file(conical_south)


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
first_run(crater_dem, crater_radius)

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0000_LROKaguyaDEM.tif'
crater_radius = 11393.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius)

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0487_LROKaguyaDEM.tif'
crater_radius = 3778.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius)


crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0048_LROKaguyaDEM.tif'
crater_radius = 570.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius)


crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0112_LROKaguyaDEM.tif'
crater_radius = 1035.0
scaling_factor = 0.5

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0204_LROKaguyaDEM.tif'
crater_radius = 6360.0
scaling_factor = 0.5

crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0229_LROKaguyaDEM.tif'
dem_detrended = '/home/nilscp/tmp/fresh_impact_craters/dem_detrended/crater0229_LROKaguyaDEM_detrended.tif'
folder = '/home/nilscp/tmp/fresh_impact_craters/shapefiles/'
crater_radius = 2575.0
scaling_factor = 0.5
first_run(crater_dem, crater_radius)



crater_dem = '/home/nilscp/tmp/fresh_impact_craters/SLDEM2015_RayedCraters/crater0112_LROKaguyaDEM.tif'
dem_detrended = '/home/nilscp/tmp/fresh_impact_craters/dem_detrended/crater0112_LROKaguyaDEM_detrended.tif'
folder = '/home/nilscp/tmp/fresh_impact_craters/shapefiles/'
(gdf_me, gdf_lm, gdf_cs, gdf_ell, gdf_centre) = load_elevations(crater_dem, folder)

'''



