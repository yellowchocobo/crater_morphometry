import copy
import os 
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg
import scipy.ndimage
from scipy.optimize import curve_fit
from scipy import optimize


#TODO change name
# could have debugging options where all of the variables produced are returned
def xy_circle(crater_radius, x_crater_center, y_crater_center):
    '''
    Parameters
    ----------
    crater_radius : float
        crater radius in the unit of the proj. coordinate system (e.g., meters).
    x_crater_center : float
        x-coordinate of the centre of the crater.
    y_crater_center : float
        y-coordinate of the centre of the crater.

    Returns
    -------
    x_circle_coord : float
        x-coordinates (512 values equally spaced over the whole circle)
    y_circle_coord : float
        y-coordinates (512 values equally spaced over the whole circle)
    '''    
    # theta goes from 0 to 2pi
    theta = np.linspace(0.0, 2*np.pi, 512) #as in Geiger et al. (2013)
        
    # compute circle coordinates
    x_circle_coord = crater_radius*np.cos(theta) + x_crater_center
    y_circle_coord = crater_radius*np.sin(theta) + y_crater_center
    
    return (x_circle_coord, y_circle_coord)

def detrending(crater_radius,
               from_scaled_radius, scaled_radius_end,
               elevations, 
               dem_resolution, 
               xmesh, ymesh, 
               x_crater_center_px, y_crater_center_px, 
               filterMedianStd):
    
    '''
    Routine to fetch all the pixel values between SCALED_RADIUS_START and 
    SCALED_RADIUS_END from the crater centre (e.g., between 2.0 R and 3.0 R
    for detrending in a regional context and 0.9 and 1.1 for detrending
    assymetries along ) 

    Parameters
    ----------
    crater_radius : float
        crater radius in the unit of the proj. coordinate system (e.g., meters).
    from_scaled_radius : float
        Distance (scaled with the radius of the crater diameter) from which the
        detrending step will be conducted. 
    scaled_radius_end : float
        Distance (scaled with the radius of the crater diameter) up to which the
        detrending step will be conducted. 
    elevations : numpy array
        Numpy array containg elevations (either original or detrended values).
    dem_resolution : float
        resolution of the DEM in meters.
    xmesh : numpy array
        mesh grid with the same dimension as the elevations.
    ymesh : numpy array
        mesh grid with the same dimension as the elevations.
    x_crater_center_px : int
        centre of the crater in pixel coordinates.
    y_crater_center_px : int
        Dcentre of the crater in pixel coordinates.
    filterMedianStd : boolean
        if True, values above the median of the elevation + one standard dev
        and below the median of the elevation - one standard dev are discarded.
        This allow 

    Returns
    -------
    detrended_elevation : numpy array 
        Detrended elevations between the specified from_scaled_radius and 
        .
        
    Suggestions for improvements
    -------
    - I am sure the selection could be made in a much more smooth way. Check
    https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle.
    This would be make this script much smaller and would avoid some unnecessary
    looping. 
    - Error message if, for some reasons, the from_scaled_radius and scaled_radius_end
    result in the fetching of elevations values outside of the mapped area.

    '''
    # in map coordinates       
    x2, y2 = xy_circle((from_scaled_radius*crater_radius) / dem_resolution, 
                       x_crater_center_px, y_crater_center_px)
    
    x3, y3 = xy_circle((scaled_radius_end*crater_radius) / dem_resolution, 
                       x_crater_center_px, y_crater_center_px)
              
    (x2, y2, x3, y3) = (np.round(x2).astype('int'), np.round(y2).astype('int'),
                        np.round(x3).astype('int'), np.round(y3).astype('int'))

    # number of values along each cross section
    # sampled at two times the dem resolution
    n_points_along_cs = np.int((crater_radius / dem_resolution) * 2.0)
    
    # empty array
    y_disk_seletion_px = []
    x_disk_seletion_px = []
    
    # Looping through all the x-coordinates of the circle located at 2R
    for i in range(len(x2)):

        # the starting coordinates move all the time and correspond to the 
        # boundary of the 2R circle
        x_cs_start = x2[i]
        y_cs_start = y2[i]
        
        # the end coordinates correspond to the boundary of the 3R circle
        x_cs_end = x3[i]
        y_cs_end = y3[i]
                
        # the distance is calculated, should be equal to two times the radius
        (x_coord_cs, y_coord_cs) = (np.linspace(x_cs_start, x_cs_end, n_points_along_cs), 
                        np.linspace(y_cs_start, y_cs_end, n_points_along_cs))
        
        # only integer here
        (x_coord_cs, y_coord_cs) = (np.round(x_coord_cs).astype('int'), 
                                    np.round(y_coord_cs).astype('int'))
        
        #need to get rid of repetitions
        rep = np.zeros((len(x_coord_cs),2))
        rep[:,0] = x_coord_cs
        rep[:,1] = y_coord_cs
        __, index = np.unique(["{}{}".format(ix, j) for ix,j in rep], 
                              return_index=True)
        
        for i in index:
            x_disk_seletion_px.append(x_coord_cs[i])
            y_disk_seletion_px.append(y_coord_cs[i])
    
    
    # these correspond to all coordinates between the slice of 2R and 3R                
    x_disk_seletion_px = np.array(x_disk_seletion_px)
    y_disk_seletion_px = np.array(y_disk_seletion_px)
           
    # elevations are extracted for the map coordinates
    z = elevations[x_disk_seletion_px,y_disk_seletion_px]
 
    # and x- and y-coordinates in meters (this is correct now) # weird
    x_disk_seletion = xmesh[x_disk_seletion_px,y_disk_seletion_px]
    y_disk_seletion = ymesh[x_disk_seletion_px,y_disk_seletion_px]
          
    # the detrending routine is used (check that again)
    Z = elevation_plane(x_disk_seletion, y_disk_seletion, 
                           z, 
                           xmesh, ymesh, 
                           filterMedianStd)
    
    # the detrended linear plane is substracted to the data
    detrended_elevation = elevations - Z
    
    return (detrended_elevation)

# changed name from linear3Ddetrending to elevation_plane
def elevation_plane(x_disk_seletion, y_disk_seletion, 
                    z, 
                    xmesh, ymesh, 
                    filterMedianStd):
    '''
    Based on the regional elevations

    Parameters
    ----------
    x_disk_seletion : numpy array
        1-D numpy array containing all the x-coordinates of cells in between
        the from_scaled_radius and scaled_radius_end in the detrending function.
    y_disk_seletion : numpy array
        1-D numpy array containing all the y-coordinates of cells in between
        the from_scaled_radius and scaled_radius_end in the detrending function.
    z : numpy array
        Elevations values within the disk of selected values. 
    xmesh : numpy array
        mesh grid with the same dimension as the elevations.
    ymesh : numpy array
        mesh grid with the same dimension as the elevations.
    filterMedianStd : boolean
        if True, values above the median of the elevation + one standard dev
        and below the median of the elevation - one standard dev are discarded.
        This allow 
        DESCRIPTION.

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
        x_filtered = x_disk_seletion[zidx]
        y_filtered = y_disk_seletion[zidx]
        z_filtered = z[zidx]
        
        # Remove values that are more than one median +-std away
        cloud_points = np.c_[x_filtered,y_filtered,z_filtered]
        
    else:
        # we don't filter values out of the selection 
        cloud_points = np.c_[x_disk_seletion,y_disk_seletion,z]
       
    # best-fit linear plane (1st-order)
    A = np.c_[cloud_points[:,0], cloud_points[:,1], np.ones(cloud_points.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, cloud_points[:,2])    # coefficients
        
    # evaluate it on the original mesh grid
    Z_plane = C[0]*xmesh + C[1]*ymesh + C[2]
    
    return (Z_plane)

def local_minima2D(array):
    '''
    Find local minimas across a 2-D numpy array.

    Parameters
    ----------
    array : 2-D numpy array

    Returns
    -------
    numpy mask array of local minimas (True/False)

    '''
    
    return ((array <= np.roll(array, 1, 0)) &
            (array <= np.roll(array, -1, 0)) &
            (array <= np.roll(array, 1, 1)) &
            (array <= np.roll(array, -1, 1)))

def local_minima(array):
    '''
    Find local minimas across a 1-D numpy array.

    Parameters
    ----------
    array : 1-D numpy array

    Returns
    -------
    numpy mask array of local minimas (True/False)

    '''
    
    return ((array >= np.roll(array, 1)) &
            (array >= np.roll(array, -1)))

def maximum_elevation(z, n_points_along_cs, cols, rows, xmesh, ymesh):
    '''
    Find the maximum elevation along a cross section (1-D numpy array).

    Parameters
    ----------
    z : 1-D numpy array
        Elevations along a cross-section.
    n_points_along_cs : int
        Number of sampled points along the cross-section.
    cols : int
        all positions (in terms of indexes in the array) across the x-axis.
    rows : int
        all positions (in terms of indexes in the array) across the y-axis.
    xmesh : numpy array
        mesh grid with the same dimension as the elevations.
    ymesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    col_coord : numpy array
        position of the maximum_elevation in DEM map projection accross the x-axis.
    row_coord : numpy array
        position of the maximum_elevation in DEM map projection accross the y-axis.
    col_px : numpy array
        position of the maximum_elevation in px coordinates accross the x-axis.
    row_px : numpy array
        position of the maximum_elevation in px coordinates accross the y-axis.
    maximum_elevation : float
        maximum elevation along the cross-section profile.

    '''    
    max_elv_idx = np.nanargmax(z)
        
    # we do not want to take the two last values (as it will likely not represent
    # the rim but the ridge/ambient topography)
    if max_elv_idx >= n_points_along_cs - 2: 
        col_coord = np.nan
        row_coord = np.nan
        
        maximum_elevation = np.nan
        
        col_px = np.nan
        row_px = np.nan
        
    # we do not want to take the values at the centre of the crater 
    # if for some reasons it should detect the higher elevation there
    # (I run into an example where it was like that)
    elif max_elv_idx == 0:
        col_coord = np.nan
        row_coord = np.nan
        
        maximum_elevation = np.nan
        
        col_px = np.nan
        row_px = np.nan
        
    # else we just take the x- and y-coordinates and the elevation of the 
    # maximum elevations
    else:        
        maximum_elevation = z[max_elv_idx]
        
        col_px = int(cols[max_elv_idx])
        row_px= int(rows[max_elv_idx])
        
        col_coord = xmesh[col_px,row_px]
        row_coord = ymesh[col_px,row_px]
        
        
    return (col_coord, row_coord, col_px, row_px, maximum_elevation)

# changed name from local_elevation to local_elevations
# search_ncells is added so that not too many local minimas are detected
def local_elevations(z, n_points_along_cs, search_ncells, 
                     cols, rows, 
                     cs_id, 
                     xmesh, ymesh):
    '''
    

    Parameters
    ----------
    z : numpy 1-D array
        Elevations along the cross-section profile.
    n_points_along_cs : int
        Number of elevations along the cross section profile.
    search_ncells : int
        Number of cells to search before and after a local minima is found.
    cols : numpy 1-D array (int)
        x-coordinate of all the elevations in the cross-section (in pixel).
    rows : numpy 1-D array (int)
        y-coordinate of all the elevations in the cross-section (in pixel).
    cs_id : int
        cross-section id.
    xmesh : numpy array
        mesh grid with the same dimension as the elevations.
    ymesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    col_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the x-axis).
    row_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the y-axis).
    col_cells_LE : numpy 1-D array (int)
        positions of the local elevations in the DEM in pixels (across the x-axis).
    row_cells_LE : numpy 1-D array (int)
        positions of the local elevations in the DEM in pixels (across the y-axis).
    elev_LE : numpy 1-D array (float)
        local elevations along the cross-section profile.
    prof_LE : int
        Several local elevations will be found per cross-section. We have thus
        the need to track the cross-section id.
    '''
    
    # create empty arrays for both map and pixel coordinates of the location of
    # local mininimas as well as their elevations and the cross-section profile 
    # number 
    col_coord_LE = np.array([])
    row_coord_LE = np.array([])
    col_cells_LE = np.array([])
    row_cells_LE = np.array([])
    elev_LE = np.array([])
    prof_LE = np.array([])

    # need first to calculate the locations of the local minima
    idx_local_minina = np.where(local_minima(z) == True)[0]
    
    #TODO - This could be replaced so that a looping is not needed.
    # get rid of the first and last values in case
    idx_local_minina_filtered = np.array([])
    
    for i in idx_local_minina:
        if i == n_points_along_cs - 1:
            None
        elif i == 0:
            None
        else:
            idx_local_minina_filtered = np.append(idx_local_minina_filtered, i)
            
    idx_local_minina_filtered = idx_local_minina_filtered.astype('int')
    
    # we do not want to have too many local minima per cross-section. In order
    # to work around this problem, for each potential local minima, we look
    # at all elevation values within 0.1*R. If the local minima is still the 
    # largest elevation value, we add it to the list of local minima for the 
    # cross section.    
    for i in idx_local_minina_filtered:
        
        # if the number of cells from the detected local minima allows for it
        # take the highest elevation within 0.1R
        if (((i - search_ncells) >= 0) & ((i+search_ncells+1 < n_points_along_cs))):
            
            # maximum elevation value towards the crater centre (within 0.1R)
            max_z_towards_cc = np.nanmax(z[i-search_ncells:i])
            
            # maximum elevation value towards the rim (within 0.1R) 
            max_z_towards_rim = np.nanmax(z[i+1:i+search_ncells+1])
            
            # check if this is larger than the local elevation
            isTrue = np.logical_and(z[i] > max_z_towards_cc, z[i] > max_z_towards_rim)
            
            # if yes save both map and pixel coordinates
            if isTrue :
                c = int(cols[i])
                r = int(rows[i])
                
                col_coord_LE = np.append(col_coord_LE, xmesh[c,r]) 
                row_coord_LE = np.append(row_coord_LE, ymesh[c,r])
                col_cells_LE = np.append(col_cells_LE, c)
                row_cells_LE = np.append(row_cells_LE, r)                                        
                elev_LE = np.append(elev_LE, z[i])
                prof_LE = np.append(prof_LE, cs_id)
                
    return (col_coord_LE, row_coord_LE, 
            col_cells_LE, row_cells_LE, 
            elev_LE, prof_LE)

    
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
    return array[idx], idx


def slope_change(zi, dist, slopenum, mingrade, clustersz, cols, rows, xc, yc):
    
    
    '''
    (col_coord_BS, row_coord_BS, col_cells_BS, row_cells_BS, elev_BS, prof_BS) = (
    slope_change(zi, dist, slopenum, mingrade, clustersz, cols, rows, xe, ye))
    
    OK 18.10.2018
    '''
    
    # create empty arrays
    slopebe4 = np.zeros(len(zi))
    slopeaft = np.zeros(len(zi))
    
    # we need to test if these are the largest values within 0.1*R of the value
    for t in np.arange(slopenum,len(dist)-slopenum,1):
        
        # find s and z value sto construct a slope
        sbe4 = np.array([dist[(t-slopenum):t], np.ones(slopenum)]).T
        zbe4 = zi[(t-slopenum):t]
        saft = np.array([dist[t:(t+slopenum)], np.ones(slopenum)]).T
        zaft = zi[t:(t+slopenum)]
        
        #need to get rid of Nans (don't need this step)
        # bug here 10.09.2018
        be4stat = np.linalg.lstsq(sbe4,zbe4)[0]
        aftstat = np.linalg.lstsq(saft,zaft)[0]
        
        slopebe4[t] = be4stat[0]
        slopeaft[t] = aftstat[0]
        
    # substract to find max slope differences
    difs = (slopebe4 - slopeaft)
    
    #make empty variables
    cluster = []
    pind = []
    
    # make empty variables to fill in for loop
    for val in difs:
        if val > mingrade:
            cluster.append(val)
            
        # once value goes below 0, cluster ends
        if val <= 0:
            if len(cluster) > clustersz:
                cluster = np.array(cluster)
                pind = (np.nonzero(difs == np.max(cluster))[0][0])
                
                # I guess they want to get the coordinates
                #if ~np.isnan(zi[pind]):
                #    inds.append(pind)
                    
            cluster = []
        
    if pind:
        ct = int(cols[pind])
        rt = int(rows[pind])
        col_coord_BS = xc[ct,rt]
        row_coord_BS = yc[ct,rt]
        col_cells_BS = ct
        row_cells_BS = rt
        elev_BS = zi[pind]
        
    else:
        col_coord_BS = np.nan
        row_coord_BS = np.nan
        col_cells_BS = np.nan
        row_cells_BS = np.nan
        elev_BS = np.nan     
        
    return col_coord_BS, row_coord_BS, col_cells_BS, row_cells_BS, elev_BS


'''
******************************************************************************
'''

def rim(xc, yc, ncenterx, ncentery, ndata, radius, cellsize, 
        slen, minclust, mingrade, first_run):
    
    '''
    OK: 18.10.2018
    '''
    
    #-- Extract the line...
    # Make a line with "num" points...
    
    # same as we have a square DEM but if not (second run, what should I do?)
    #ye = xe
    
    centerx = ncenterx
    centery =  ncentery # nunber of rows and columns are the same
    
    # map coordinates x2, y2
    x2c, y2c = xy_circle((2.0*radius) / cellsize, ncenterx, ncentery)
    
    # real values x2, y2
    (x2c, y2c) = (np.round(x2c).astype('int'), np.round(y2c).astype('int'))
                
    # and x- and y-coordinates in meters (this is correct now)
    #x2 = xc[x2c,y2c]
    #y2 = yc[x2c,y2c]
    
    # we define the ME variables
    col_coord_ME = np.ones(len(x2c))
    row_coord_ME = np.ones(len(x2c))
    col_cells_ME = np.ones(len(x2c))
    row_cells_ME = np.ones(len(x2c))
    elev_ME = np.ones(len(x2c))
    prof_ME = np.arange(0,512)
    
    # we define the LE variables
    col_coord_LE = np.array([])
    row_coord_LE = np.array([])
    col_cells_LE = np.array([])
    row_cells_LE = np.array([])
    elev_LE = np.array([])
    prof_LE = np.array([])
    
    # we define the break in slope variables
    col_coord_BS = np.ones(len(x2c))
    row_coord_BS = np.ones(len(x2c))
    col_cells_BS = np.ones(len(x2c))
    row_cells_BS = np.ones(len(x2c))
    elev_BS = np.ones(len(x2c))
    prof_BS = np.arange(0,512)
    
    
    # set arrays equal to nan
    col_coord_ME[col_coord_ME == 1] = np.nan
    row_coord_ME[row_coord_ME == 1] = np.nan
    col_cells_ME[col_cells_ME == 1] = np.nan
    row_cells_ME[row_cells_ME == 1] = np.nan
    elev_ME[elev_ME == 1] = np.nan
    
    col_coord_BS[col_coord_BS == 1] = np.nan
    row_coord_BS[row_coord_BS == 1] = np.nan
    col_cells_BS[col_cells_BS == 1] = np.nan
    row_cells_BS[row_cells_BS == 1] = np.nan
    elev_BS[elev_BS == 1] = np.nan

    # samples at half the cellsize 
    num = np.int(np.ceil(2.0*radius/cellsize)*2.0)
    
    # calculate the threshold
    thres = 0.1 * radius
    ncells = np.int(np.ceil(thres/cellsize))
    
    # find out how many points will be in each slope segment
    slopenum = int(num*slen) #slen = 0.1
    clustersz = num*minclust #minclust = 0.05 is it used some where?
    
    for ix in range(len(x2c)):

        # find the map coordinates (xe and ye are the same)     
        ncol = x2c[ix]
        nrow = y2c[ix]
                
        # the distance is calculated, should be equal to two times the radius
        cols, rows = np.linspace(centerx, ncol, num), np.linspace(centery, nrow, num)
        
        # Extract the values along the line, using cubic interpolation and the 
        # map coordinates
        zi = scipy.ndimage.map_coordinates(ndata, np.vstack((cols,rows)))
        
        # calculate the distance along the profile
        dist_cells = np.sqrt(((cols - centerx)**2.) + ((rows - centery)**2.))
        dist = dist_cells * cellsize #I guess it is what they call s in Geiger
        
        '''
        From Geiger 2013: If the ME occurs at the last point on the profile, 
        it is not recorded for that profile, to avoid selecting a rim location 
        on a local rise or ridge.
        '''
        # Maximum elevation     
        (col_coord_ME[ix], row_coord_ME[ix], col_cells_ME[ix], row_cells_ME[ix], 
         elev_ME[ix]) = maximum_elevation(zi, num, cols, rows, xc, yc)
        
        if first_run:
            
            None
        
        else:
        
            # Local elevation
            (col_coord_LE_tmp, row_coord_LE_tmp, col_cells_LE_tmp, 
             row_cells_LE_tmp, elev_LE_tmp, prof_LE_tmp) = local_elevation(zi, 
                                                       num, ncells, cols, rows, ix, 
                                                       xc, yc)
            
            col_coord_LE = np.append(col_coord_LE, col_coord_LE_tmp)
            row_coord_LE = np.append(row_coord_LE, row_coord_LE_tmp)
            col_cells_LE = np.append(col_cells_LE, col_cells_LE_tmp)
            row_cells_LE = np.append(row_cells_LE, row_cells_LE_tmp)
            elev_LE = np.append(elev_LE, elev_LE_tmp)
            prof_LE = np.append(prof_LE, prof_LE_tmp)
            
            
            # break in slopes
            (col_coord_BS[ix], row_coord_BS[ix], col_cells_BS[ix], 
             row_cells_BS[ix], elev_BS[ix]) = (slope_change(zi, dist, slopenum, 
                                  mingrade, clustersz, cols, rows, xc, yc))

    if first_run:
        
        return (col_coord_ME, row_coord_ME, col_cells_ME, row_cells_ME, elev_ME, prof_ME)
    
    else:
            
        return (col_coord_ME, row_coord_ME, col_cells_ME, row_cells_ME, elev_ME, prof_ME,
                col_coord_LE, row_coord_LE, col_cells_LE, row_cells_LE, elev_LE, prof_LE,
                col_coord_BS, row_coord_BS, col_cells_BS, row_cells_BS, elev_BS, prof_BS)


'''
******************************************************************************
''' 

def linear(x,a,b):
    y = a*x + b
    return y

'''
******************************************************************************
'''
def delete_redundant(xcoord_detrend, ycoord_detrend, 
                                   elev_detrend, prof_detrend):
    
    '''
    testx, testy, testz, testprof = delete_redundant(xcoord_ME, ycoord_ME, 
                                   elev_ME, profile_ME)
    '''
    
    uni = np.ones(len(xcoord_detrend))
    for ii, val in np.ndenumerate(xcoord_detrend):
        iii = ii[0]
        xcoord_tmp = xcoord_detrend[iii]
        ycoord_tmp = ycoord_detrend[iii]
        
        ix_tmp = np.where(np.logical_and(xcoord_detrend == xcoord_tmp, 
                                         ycoord_detrend == ycoord_tmp))
        uni[iii] = ix_tmp[0][0]
        
    unique_ix = np.unique(uni)
    unique_ix = unique_ix.astype('int')
    
    return (xcoord_detrend[unique_ix], ycoord_detrend[unique_ix], 
            elev_detrend[unique_ix], prof_detrend[unique_ix])
    
    
    
'''
******************************************************************************
''' 

def rim_composite(col_coord_ME, row_coord_ME, col_cells_ME,
                  row_cells_ME, elev_ME, profile_ME, colint, rowint, colmap, rowmap,
                  xc, yc,
                  elevint, profint, ncenterx, ncentery,
                  angle, stangle, Drad, Dint, 
                  contloop, siftRedundant, kpstitch):
            
    '''
    
    
    # Maximum allowed radial discontinuity Drad (I should convert these values in cells)
    Drad = 0.1 * r
    
    # Distance of interest (searching distance)
    Dint = 0.05 * r
    
    # Maximum angular discontinuity (avoid unnecessary large gap angle in the 
    # data)
    angle = 2.0 #(in degrees)
    
    stangle = [0,45,90,135,180,225,270,315]
    
    contloop = True
    siftRedundant = True
    kpstitch = False
    
    OptRims, Omegas, gap, maxradf = rim_composite(col_coord_ME, row_coord_ME, col_cells_ME,
                                                  row_cells_ME, elev_ME, profile_ME,
                                                  colint, rowint, colmap, 
                                                  rowmap, elevint, profint,
                                                  angle, stangle, Drad, Dint, 
                                                  contloop, siftRedundant, kpstitch)
    
    
    plt.pcolor(xc, yc, ndata2)
    plt.colorbar()
    plt.plot(x1,y1,"b")
    plt.plot(OptRims[0][0,:],OptRims[0][1,:],"ko")
    plt.plot(OptRims[1][0,:],OptRims[1][1,:],"ro")
    plt.plot(colint, rowint, "yo")
    '''    
    lMat = len(col_coord_ME)
    
    # converted from degrees to our circle divided in 512 radial profiles
    angle = np.ceil(angle* (lMat/360.))
    
    # converts from 360 degrees to a 512 radial profile circle
    stpnts = (lMat * np.array(stangle))/(360.)
    stpnts = stpnts.astype('int')
    
    # defining empty variables
    GMR = np.zeros(lMat)
    IndGM = np.zeros((2,lMat))
    IndLMR = np.zeros((2,len(colint)))
    OptRims = []
    Omegas = []
    Lilo = []
    maxradf = []
    gaplist = []
    
    '''
    ***********************MAXIMUM ELEVATION**************************************
    '''
    #Only work with not nan values (previous nan-values will be equal to 0)
    nnan = np.where(np.isfinite(col_cells_ME))
    
    # calculate the radius to the global maximum elevation
    ab = (row_coord_ME[nnan] - yc[ncenterx,ncentery])**2.0 #changed 
    bc = (col_coord_ME[nnan] - xc[ncenterx,ncentery])**2.0 #changed
    GMR[nnan] = np.sqrt(ab + bc)
    
    # get the indices of the global maximum elevation
    IndGM[0,[nnan]] = col_cells_ME[nnan]
    IndGM[1,[nnan]] = row_cells_ME[nnan]
    IndGM =IndGM.astype('int')
    
    '''
    ***********************LOCAL ELEVATION**************************************
    '''
    
    # calculate the radius to the global maximum elevation
    ab = (rowint - yc[ncenterx,ncentery])**2.0 #changed
    bc = (colint - xc[ncenterx,ncentery])**2.0 #changed
    LMR = np.sqrt(ab + bc)
    
    # get the indices of the global maximum elevation
    IndLMR[0,:] = colmap
    IndLMR[1,:] = rowmap
    IndLMR =IndLMR.astype('int')
    
    '''
    ***********************LOOPS **************************************
    '''
    
    for strt in stpnts:
                
        #counter clockwise
        ccw = np.concatenate((np.arange(strt,lMat),np.arange(0,strt)))
        
        # clockwise
        cw = np.concatenate(((np.arange(strt+1)[::-1]),np.arange(strt+1,lMat)[::-1]))
        
        # take both loops
        loops = [cw, ccw]
        
        
        # count the number of counter clockwise and clockwise loops
        pnum = 0 # for a stangle = [0, 90, 180, 270], we should have 8 rims
        
        for path in loops:
            
            #print 'On path ' + str(strt) + ' ' + str(pnum)
            
            #create empty rim trace for this path
            
            RIM = np.zeros([5,lMat]) #that will contain the future CRT X, Y and Z and choi? np.zeros([3,lMat])
            Stitch = [] #profiles that will be skipped
            maxrad = []
            
            # Find last point of loop to have reference for start
            # We need to find the maximum elevation for this profile
            LastR = GMR[path[-1]]
            Lind = IndGM[:,path[-1]]
            
            # in case there are no maximum elevation in this profile
            
            # search in the profile before until it find a value
            uhoh = -2
            
            while (LastR == 0):
                LastR = GMR[path[uhoh]]
                Lind = IndGM[:,path[uhoh]]
                uhoh -=1
                if np.abs(uhoh) >= lMat:
                    break
                    
                #I had an error where no LastR have been found
                #it loops until -513 (which is out of bounds)
                
            # define some variables to start with
            Om = 0
            k = 0
            gap = 0
            before = False
            
            #
            while before == False:
                
                # what is the index
                i = path[k]
                
                # cand and choi are resetted at the start
                cand = []
                choi = []            
                
                # is the radius of the global max at that index within reason
                ub = np.ceil(LastR + Drad)
                lb = np.floor(LastR - Drad)
                
                if np.logical_and(GMR[i] <= ub, GMR[i] >= lb):
                    
                    choi = i # this is modified
                    flag = 0
                    #cand = i #this was added (equal to the profile number)
                    
                #if not look for other interest points
                else:
                    
                    # keep track of # off the max method
                    Om += 1
                    
                    # search in local maxima, are they some candidates?
                    posI = np.where(profint == i)
                    
                    # if not (should actually not go through this loop so often)
                    if len(posI[0]) == 0:
                        
                        #print ('At ' + str(i*360/512) + ' degrees, there\'s no interest points')
                                                        
                        # assign gap and move
                        gap += 1
                        k += 1
                        
                        # if we are at the end start over
                        if k >= len(path):
                            k = 0
                        continue
                    
                    #cand = []
                    #choi = []
                    
                    # not sure about the place of this thing
                    if Dint:
                        
                        # reset angular distance
                        adis = 0
                        ak = k
                        aind = path[ak]
                        #sti = []
                        
                        while adis <= angle:
                        
                            #find possible candidate in next spoke
                            posI = np.where(profint == aind)[0]
                            
                            #get their radius
                            posR = LMR[posI] - LastR
                            
                            #find the radial distance of the candidate from laspnt
                            cand = posI[np.nonzero(np.abs(posR) <= Dint)]
                            
                            # if candidates is empty random.py will raise an erro
                            # so we have to use try
                            
                            try:
                                choi = random.choice(cand)
                                flag = 1
                                
                                # then stop the while loop
                                break
                            #if not, go to the next profile
                            except:
                                adis += 1
                                ak += 1
                                if ak == len(path):
                                    ak = 0
                                aind = path[ak]
                                
                                
                            if ((adis > 0) & (adis <= angle)): #if something was skipped
                                
                                Stitch.append(aind) #number of the profile
                                #this part is different from Geiger
                                # I don't understand what kind of x1, x2, y1, y2
                                # they put in stitch, but I guess they put the 
                                # whole radial profile
                                
                                i = aind
                                k = ak
                                
                # if it does not work just pick the closest        
                if choi == []:
                    posI = np.where(profint == i)[0]
                                                
                    #get their radius
                    posR = np.abs(LMR[posI] - LastR)
                    
                    #I guess some of the largest radial discontinuities should
                    # happen here as others should be within 0.05*r or 0.1*r
                    # in case there are no data
                    if len(posR >= 1):
                        maxrad.append(np.min(posR))
                        
                        #candidate is closest
                        cand = posI[np.nanargmin(posR)]
                        
                        # if several with the same minimal distance # don't need 
                        # with nan argmin
                        choi = cand
                        flag = 1
                    else:
                        flag = 2
                    
                # special for me
                if flag == 1:
                    xtmp = colint[choi]
                    ytmp = rowint[choi]
                    ztmp = elevint[choi]
                    
                    # Assign new last R
                    LastR = LMR[choi]
                    Lind = choi
                    
                elif flag == 0:
                    xtmp = col_coord_ME[choi]
                    ytmp = row_coord_ME[choi]
                    ztmp = elev_ME[choi]
                    #print elev_ME[choi]
                    
                    # Assign new last R
                    LastR = GMR[choi]
                    Lind = choi #I don't think I need that
                    
                else:
                    #if no values are found at all
                    xtmp = np.nan
                    ytmp = np.nan
                    ztmp = np.nan
                    
                    
                if RIM[0,i] == xtmp:
                    if RIM[1,i] == ytmp:
                        if RIM[2,i] == ztmp:
                            before = True                                    
                           
                RIM[0,i] = xtmp
                RIM[1,i] = ytmp
                RIM[2,i] = ztmp
                RIM[3,i] = path[k] # new it takes where in the profile
                RIM[4,i] = flag

                
                # increment the path loop
                k += 1
                
                # if at the end of the path start over
                if k >= len(path):
                    k = 0
                    if contloop == False:
                        before = True
                                
                                
            # when finished, divide the points off Maxmethod by total
            Om = Om / float(lMat) # (Omega)
            aprim = True
    
            # if not the first, check if            
            if len(OptRims) > 0:
                for j in range(len(OptRims)):
                    if sum(RIM[0,:] != OptRims[j][0,:]) == 0:
                        #print "Trace is same as Trace " + str(j)
                        
                        if siftRedundant:
                            aprim = False
                        break
                    
            if aprim:
                OptRims.append(copy.deepcopy(RIM))
                Omegas.append(Om)
                gaplist.append(gap) #number of gap
                maxrad = np.array(maxrad)
                maxrad_sorted = maxrad[np.argsort(maxrad)][-5:]
                maxradf.append(maxrad_sorted)
                
                if kpstitch:
                    Stitch = np.array(Stitch)
                    Lilo.append(Stitch)
            pnum +=1
                
    if kpstitch:
        return OptRims, Omegas, Lilo, gaplist, maxradf
    else:
        return OptRims, Omegas, gaplist, maxradf

'''
******************************************************************************
''' 

def leastsq_circle(x,y):
    
    # should only take not nan values
    
    # coordinates of the barycenter
    
    xt = x[~np.isnan(x)]
    yt = y[~np.isnan(y)]
    
    x_m = np.nanmean(xt)
    y_m = np.nanmean(yt)
    center_estimate = x_m, y_m
    
    center, ier = optimize.leastsq(f, center_estimate, args=(xt,yt))    
    xc, yc = center
    Ri       = calc_R(xt, yt, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

'''
******************************************************************************
''' 

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

'''
******************************************************************************
''' 

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


'''
******************************************************************************
''' 

def plot_data_circle(x,y, xc, yc, R):
    f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-np.pi, np.pi, 180)

    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'bo' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')   
    # plot data
    plt.plot(x, y, 'ro', label='data', mew=1)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    
'''
******************************************************************************
'''

def power(x,a,b,c):
    
    return a * (x**b) + c

'''
******************************************************************************
'''

def calculation(xc, yc, x_not_outliers, y_not_outliers, z_not_outliers, prof_not_outliers,
                Rn, ndata, cellsize, ncenterx, ncentery, xllcorner, yllcorner):
    
    '''
       
    xe = np.arange(cellsize/2.,(ncols*cellsize),cellsize)
    ye = np.arange(cellsize/2.,(nrows*cellsize),cellsize)
    
    Rn = rnew

    x2, y2 = wk.xy_circle(2.0*Rn, xe[ncols/2], ye[ncols/2]) # or extracting the exact coordinates
                                                         # x, y (substracting for the lower and upper corner)
    xcn = xe[ncols/2]
    ycn = ye[nrows/2]                                                     
    

    
    
    '''
    #-- Extract the line...
    # Make a line with "num" points...
    
        
    # Find unique indices from detected (other way)
    idx_detected1 = np.zeros((len(x_not_outliers),2))
    idx_detected1[:,0] = (x_not_outliers)/cellsize
    idx_detected1[:,1] = (y_not_outliers)/cellsize
    
    # don't really need to take the integers?
    idx_detected1 = idx_detected1.astype('int')
        
    __, index = np.unique(["{}{}".format(i, j) for i,j in idx_detected1], return_index=True)
    idx_detected1_uni = idx_detected1[index,:] #I think that this work
    prof_uni_detected = prof_not_outliers[index] #still contain zeros
    
    # I should also maybe take only detected rim points within 0.9 to 1.1 rnew?
          
    
    # 2r from the center of the  crater
    idx_circle2 = np.zeros((len(idx_detected1_uni),2))
    idx_circle2[:,0] = ((idx_detected1_uni[:,0] - ncenterx)*2.)+ ncenterx    
    idx_circle2[:,1] = ((idx_detected1_uni[:,1] - ncentery)*2.) + ncentery
    
    # samples at half the cellsize 
    num = np.int(np.ceil(2.0*Rn/cellsize)*2.0)
    
    
    # I need to define all empty arrays here
    diamd = np.zeros(len(idx_circle2))
    R_upcw = np.zeros(len(idx_circle2))
    R_ufrc = np.zeros(len(idx_circle2))
    cse = np.zeros(len(idx_circle2))
    slope_mcw = np.zeros(len(idx_circle2))
    slope_ucw = np.zeros(len(idx_circle2))
    slope_fsa = np.zeros(len(idx_circle2))
    slope_lrs = np.zeros(len(idx_circle2)) #lower rim span
    slope_urs = np.zeros(len(idx_circle2)) #upper rim span
    h = np.zeros(len(idx_circle2))
    depth = np.zeros(len(idx_circle2))
    crdl = np.zeros(len(idx_circle2))
    frdl = np.zeros(len(idx_circle2))
    hr = np.zeros(len(idx_circle2))
    
    #zall = np.zeros((num,len(idx_circle2)))
    #distall= np.zeros((num,len(idx_circle2)))
    
	# maybe use delete_redundant function here? on idx_circle2?
	# I think it's okay see above at L1100-1108
    
    #get values
    idt = 0
    #depth = 0
	
	#dictionary to save the cross sections to
    crossSections = dict()
    XSections = dict()
    YSections = dict()
    
    # need to think here
    for crossi, ii in enumerate(idx_circle2):
                
        ncol = ii[0]
        nrow = ii[1]
        
        jj = idx_detected1_uni[idt]
        ncol_1r = jj[0]
        nrow_1r = jj[1]
        
        ## indices of detected rim
        #jj = idx_detected1_uni[pp[0]]
        
        #nrowd = jj[0]
        #ncold = jj[1]
           
        # the distance is calculated, should be equal to two times the radius
        cols, rows = np.linspace(ncenterx, ncol, num), np.linspace(ncentery, nrow, num)
        

        # Extract the values along the line, using cubic interpolation and the 
        # map coordinates
        zi = scipy.ndimage.map_coordinates(ndata, np.vstack((cols,rows))) # changed here 
        
        #plt.scatter(xc[cols.astype('int'),rows.astype('int')],yc[cols.astype('int'),rows.astype('int')],c=zi,s=50)
        
        #zall[:,idt] = zi
               
        # calculate the distance along the profile 2
        dist_cells = np.sqrt(((cols - ncenterx)**2.) + ((rows - ncentery)**2.))
        dist = dist_cells * cellsize #I guess it is what they call s in Geiger
		
		# I should here save each profile that could later on be used (either saved in a dictionary or 
		# directly save to a text file. I would prefer first to be saved in a dictionary and then
		# save to a text file) HERE MODIFY
        crossSections[crossi] = zi[:]
        XSections[crossi] = cols
        YSections[crossi] = rows
        
        #I would also like to save the location x, y of the data
        
        
        #distall[:,idt] = dist #this is going to be big for every profile
        
        #distance to maximum or local elevation        
        #dist1R = np.sqrt(  ((yc[ncenterx,ncentery]-yc[ncol_1r,nrow_1r])**2.) + ((xc[ncenterx,ncentery]-xc[ncol_1r,nrow_1r])**2.) )
        
        #find nearest value to the average radius (need to change that), does not work perfectly
        #value_nearest, idx_nearest = find_nearest(dist, dist1R) #closest to the actual value
        
        # ncol_1r and nrow_1r needs to be integer
        value_nearest, idx_nearest = find_nearest(zi, ndata[ncol_1r,nrow_1r])
        
        diamd[idt] = dist[idx_nearest] * 2.0
        
        #distance normalized 
        dist_norm = dist/dist[idx_nearest]
                
        # find index
        A , idxA = find_nearest(dist_norm,0.0)
        B , idxB = find_nearest(dist_norm,0.1)
        C , idxC = find_nearest(dist_norm,0.7)
        D , idxD = find_nearest(dist_norm,0.8)
        E , idxE = find_nearest(dist_norm,0.9)
        F , idxF = find_nearest(dist_norm,1.0)
        G , idxG = find_nearest(dist_norm,1.2)
        H , idxH = find_nearest(dist_norm,2.0) # should be the maximum or end of the profile
        
        '''
        **************************************************************************
        '''
        # for upper cavity-wall radius of curvature
        # radius of circle fitted to the profile from D to F
        interval_upcw = zi[idxD:idxF+1]
        dist_upcw = dist[idxD:idxF+1]
        
        # we are interest in R_upcw
        #x_upcw, y_upcw, R_upcw, residu =leastsq_circle(dist_upcw,interval_upcw)
        try:
            __, __, R_upcw[idt], __ =leastsq_circle(dist_upcw,interval_upcw)
            
        except:
            R_upcw[idt] = np.nan
        
        
        '''
        **************************************************************************
        '''
        # for upper flank radius of curvature
        interval_ufrc = zi[idxF:idxG+1]
        dist_ufrc = dist[idxF:idxG+1]
        
        #x_ufrc, y_ufrc, R_ufrc, residu =leastsq_circle(dist_ufrc,interval_ufrc)
        
        # new addition
        try:
            __, __, R_ufrc[idt], __ =leastsq_circle(dist_ufrc,interval_ufrc)
            
        except:
            R_ufrc[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # cavity shape exponent
        interval_cse = zi[idxB:idxE+1]
        dist_cse = dist[idxB:idxE+1]
        
        try:
            a, b = curve_fit(power,dist_cse,interval_cse)
                
            exponent = a[1]
            cse[idt] = exponent
            
        except:
            cse[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # middle cavity wall slope angle
        #line fitted through
        interval_mcw = zi[idxC:idxE+1]
        dist_mcw = dist[idxC:idxE+1]
        
        try:
            a, b = curve_fit(linear, dist_mcw, interval_mcw)
            xs = np.linspace(np.min(dist_mcw),np.max(dist_mcw),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_mcw[idt] = tetarad * (180./np.pi)
            
        except:
            slope_mcw[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # upper cavity wall slope angle
        interval_ucw = zi[idxD:idxF+1]
        dist_ucw = dist[idxD:idxF+1]
        
        try:
            a, b = curve_fit(linear, dist_ucw, interval_ucw)
            xs = np.linspace(np.min(dist_ucw),np.max(dist_ucw),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_ucw[idt] = tetarad * (180./np.pi)
            
        except:
            slope_ucw[idt] = np.nan
        
        
        '''
        **************************************************************************
        '''
        
        # flank slope angle
        interval_fsa = zi[idxF:idxG+1]
        dist_fsa = dist[idxF:idxG+1]             
        
        try:
            a, b = curve_fit(linear, dist_fsa, interval_fsa)
            xs = np.linspace(np.min(dist_fsa),np.max(dist_fsa),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_fsa[idt] = np.abs(tetarad * (180./np.pi))
            
            #upper and lower rim span
            slope_urs[idt] = 180. -  (slope_ucw[idt] + slope_fsa[idt])
            slope_lrs[idt] =  180. - (slope_mcw[idt] + slope_fsa[idt])
            
        except:
            slope_fsa[idt] = np.nan
            slope_urs[idt] = np.nan
            slope_lrs[idt] = np.nan
        
        '''
        **************************************************************************
        '''
    
        #average rim height
        h[idt] = zi[idxF]

        '''
        **************************************************************************
        '''        
        
        interval_hr = zi[idxF:idxH] 
        
        # minimum values beyond the rim of the crater
        
        try:
            min_h = np.nanmin(interval_hr)
            
            # height from the rim to the smallest elevation beyond the rim of the  crater 
            hr[idt] = zi[idxF] - min_h
            
        except:
            hr[idt] = np.nan
        
        '''
        **************************************************************************
        '''        
        # calculate the depth (new way where the min along each cross section is taken)
        depth_tmp = np.min(zi)
        depth[idt] = depth_tmp
        
        '''
        **************************************************************************
        '''
        
        # flank rim decay length
        interval_frdl = zi[idxF:idxH] # I think I can not use idxH+1 otherwise it is outside
        dist_frdl = dist[idxF:idxH] # The question is the rim flank going all the way up to 2 radius?
        
        try:
            
            frdlx1 = dist_frdl[:-1]
            frdlx2 = dist_frdl[1:]
            frdly1 = interval_frdl[:-1]
            frdly2 = interval_frdl[1:]
            
            dx = frdlx2 - frdlx1
            dy = frdly2 - frdly1
            
            tetarad = np.arctan(dy/dx)
            slope_frdl = np.abs(tetarad * (180./np.pi))
            
            # get the maximum slope
            slope_frdl_max = np.nanmax(slope_frdl)
            
            # where it is the closest of half the maximum
            __, idx_frdl = find_nearest(slope_frdl, slope_frdl_max/2.0)
            
            # get the distance at half the maximum
            frdl[idt] = dist_frdl[idx_frdl] - dist[idxF]
            
        except:
            frdl[idt] = np.nan
            
        '''
        **************************************************************************
        '''
            
        # cavity rim decay length
        interval_crdl = zi[idxA:idxF+1] # I think I can not use idxH+1 otherwise it is outside
        dist_crdl = dist[idxA:idxF+1]
        
        try:
            crdlx1 = dist_crdl[:-1]
            crdlx2 = dist_crdl[1:]
            crdly1 = interval_crdl[:-1]
            crdly2 = interval_crdl[1:]
            
            dx = crdlx2 - crdlx1
            dy = crdly2 - crdly1
            
            tetarad = np.arctan(dy/dx)
            slope_crdl = np.abs(tetarad * (180./np.pi))
            
            # get the maximum slope
            slope_crdl_max = np.nanmax(slope_crdl)
            
            # where it is the closest of half the maximum
            __, idx_crdl = find_nearest(slope_crdl, slope_crdl_max/2.0)
            
            # get the distance at half the maximum
            crdl[idt] = dist[idxF] - dist_crdl[idx_crdl]
            
            # volume gets calculate afterwards from calculate_volume.py
            
        except:
            crdl[idt] = np.nan
        
        idt = idt + 1
    
        '''
        **************************************************************************
        '''
        
        
    return (R_upcw, R_ufrc, cse, slope_mcw, slope_ucw, slope_fsa, slope_lrs, slope_urs, crdl, frdl,
            h, hr, depth, diamd, len(idx_circle2), prof_uni_detected, crossSections, YSections, XSections)



'''
******************************************************************************
'''

def calculation_alt(xc, yc, x_not_outliers, y_not_outliers, z_not_outliers, prof_not_outliers,
                Rn, ndata, cellsize, ncenterx, ncentery, xllcorner, yllcorner):
    
    '''
    03.05.2019: 
    I was trying here to find an alternative to only using the "detected" crater rim. 
    The problem is that for profiles where the rim is not detected, calculations for the
    different morphological parameters (which, depends on the accurate detection of the crater rim)
    will not work...

    I was looking if I could take simply the newly calculated radius of the crater.                                                  
    

    
    
    '''
    #-- Extract the line...
    # Make a line with "num" points...
    
    # get new circle
    xnewcenter, ynewcenter, rnew, residu = leastsq_circle(x_not_outliers,y_not_outliers)
    
    # one radius
    x1, y1 = xy_circle(1.0*rnew, xnewcenter, ynewcenter)
    x1_idx = x1 / cellsize
    y1_idx = y1 / cellsize
    
    # two radius
    x2, y2 = xy_circle(2.0*rnew, xnewcenter, ynewcenter)
    x2_idx = x2 / cellsize
    y2_idx = y2 / cellsize
    
    # Find unique indices from detected (other way), does it have to be an integer?
    idx_circle2 = np.zeros((len(x2_idx),2))
    idx_circle2[:,0] = x2_idx
    idx_circle2[:,1] = y2_idx
    
    # samples at half the cellsize 
    num = np.int(np.ceil(2.0*Rn/cellsize)*2.0)
    
    
    # I need to define all empty arrays here
    diamd = np.zeros(len(idx_circle2))
    R_upcw = np.zeros(len(idx_circle2))
    R_ufrc = np.zeros(len(idx_circle2))
    cse = np.zeros(len(idx_circle2))
    slope_mcw = np.zeros(len(idx_circle2))
    slope_ucw = np.zeros(len(idx_circle2))
    slope_fsa = np.zeros(len(idx_circle2))
    slope_lrs = np.zeros(len(idx_circle2)) #lower rim span
    slope_urs = np.zeros(len(idx_circle2)) #upper rim span
    h = np.zeros(len(idx_circle2))
    depth = np.zeros(len(idx_circle2))
    crdl = np.zeros(len(idx_circle2))
    frdl = np.zeros(len(idx_circle2))
    
    #zall = np.zeros((num,len(idx_circle2)))
    #distall= np.zeros((num,len(idx_circle2)))
    
	# maybe use delete_redundant function here? on idx_circle2?
	# I think it's okay see above at L1100-1108
    
    #get values
    idt = 0
    #depth = 0
	
	#dictionary to save the cross sections to
    crossSections = dict()
    XSections = dict()
    YSections = dict()
    
    # need to think here
    for crossi, ii in enumerate(idx_circle2):
                
        ncol = ii[0]
        nrow = ii[1]
        
        # for estimating the median diameter
        jj = idx_detected1_uni[idt]
        ncol_1r = jj[0]
        nrow_1r = jj[1]
        
        ## indices of detected rim
        #jj = idx_detected1_uni[pp[0]]
        
        #nrowd = jj[0]
        #ncold = jj[1]
           
        # the distance is calculated, should be equal to two times the radius
        cols, rows = np.linspace(ncenterx, ncol, num), np.linspace(ncentery, nrow, num)
        

        # Extract the values along the line, using cubic interpolation and the 
        # map coordinates
        zi = scipy.ndimage.map_coordinates(ndata, np.vstack((cols,rows))) # changed here 
        
        #plt.scatter(xc[cols.astype('int'),rows.astype('int')],yc[cols.astype('int'),rows.astype('int')],c=zi,s=50)
        
        #zall[:,idt] = zi
               
        # calculate the distance along the profile 2
        dist_cells = np.sqrt(((cols - ncenterx)**2.) + ((rows - ncentery)**2.))
        dist = dist_cells * cellsize #I guess it is what they call s in Geiger
		
		# I should here save each profile that could later on be used (either saved in a dictionary or 
		# directly save to a text file. I would prefer first to be saved in a dictionary and then
		# save to a text file) HERE MODIFY
        crossSections[crossi] = zi[:]
        XSections[crossi] = cols
        YSections[crossi] = rows
        
        #I would also like to save the location x, y of the data
        
        
        #distall[:,idt] = dist #this is going to be big for every profile
        
        #distance to maximum or local elevation        
        #dist1R = np.sqrt(  ((yc[ncenterx,ncentery]-yc[ncol_1r,nrow_1r])**2.) + ((xc[ncenterx,ncentery]-xc[ncol_1r,nrow_1r])**2.) )
        
        #find nearest value to the average radius (need to change that), does not work perfectly
        #value_nearest, idx_nearest = find_nearest(dist, dist1R) #closest to the actual value
        
        #ok
        value_nearest, idx_nearest = find_nearest(zi, ndata[ncol_1r,nrow_1r])
        
        diamd[idt] = dist[idx_nearest] * 2.0
        
        #distance normalized 
        dist_norm = dist/dist[idx_nearest]
                
        # find index
        A , idxA = find_nearest(dist_norm,0.0)
        B , idxB = find_nearest(dist_norm,0.1)
        C , idxC = find_nearest(dist_norm,0.7)
        D , idxD = find_nearest(dist_norm,0.8)
        E , idxE = find_nearest(dist_norm,0.9)
        F , idxF = find_nearest(dist_norm,1.0)
        G , idxG = find_nearest(dist_norm,1.2)
        H , idxH = find_nearest(dist_norm,2.0) # should be the maximum or end of the profile
        
        '''
        **************************************************************************
        '''
        # for upper cavity-wall radius of curvature
        # radius of circle fitted to the profile from D to F
        interval_upcw = zi[idxD:idxF+1]
        dist_upcw = dist[idxD:idxF+1]
        
        # we are interest in R_upcw
        #x_upcw, y_upcw, R_upcw, residu =leastsq_circle(dist_upcw,interval_upcw)
        try:
            __, __, R_upcw[idt], __ =leastsq_circle(dist_upcw,interval_upcw)
            
        except:
            R_upcw[idt] = np.nan
        
        
        '''
        **************************************************************************
        '''
        # for upper flank radius of curvature
        interval_ufrc = zi[idxF:idxG+1]
        dist_ufrc = dist[idxF:idxG+1]
        
        #x_ufrc, y_ufrc, R_ufrc, residu =leastsq_circle(dist_ufrc,interval_ufrc)
        
        # new addition
        try:
            __, __, R_ufrc[idt], __ =leastsq_circle(dist_ufrc,interval_ufrc)
            
        except:
            R_ufrc[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # cavity shape exponent
        interval_cse = zi[idxB:idxE+1]
        dist_cse = dist[idxB:idxE+1]
        
        try:
            a, b = curve_fit(power,dist_cse,interval_cse)
                
            exponent = a[1]
            cse[idt] = exponent
            
        except:
            cse[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # middle cavity wall slope angle
        #line fitted through
        interval_mcw = zi[idxC:idxE+1]
        dist_mcw = dist[idxC:idxE+1]
        
        try:
            a, b = curve_fit(linear, dist_mcw, interval_mcw)
            xs = np.linspace(np.min(dist_mcw),np.max(dist_mcw),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_mcw[idt] = tetarad * (180./np.pi)
            
        except:
            slope_mcw[idt] = np.nan
        
        '''
        **************************************************************************
        '''
        
        # upper cavity wall slope angle
        interval_ucw = zi[idxD:idxF+1]
        dist_ucw = dist[idxD:idxF+1]
        
        try:
            a, b = curve_fit(linear, dist_ucw, interval_ucw)
            xs = np.linspace(np.min(dist_ucw),np.max(dist_ucw),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_ucw[idt] = tetarad * (180./np.pi)
            
        except:
            slope_ucw[idt] = np.nan
        
        
        '''
        **************************************************************************
        '''
        
        # flank slope angle
        interval_fsa = zi[idxF:idxG+1]
        dist_fsa = dist[idxF:idxG+1]             
        
        try:
            a, b = curve_fit(linear, dist_fsa, interval_fsa)
            xs = np.linspace(np.min(dist_fsa),np.max(dist_fsa),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_fsa[idt] = np.abs(tetarad * (180./np.pi))
            
            #upper and lower rim span
            slope_urs[idt] = 180. -  (slope_ucw[idt] + slope_fsa[idt])
            slope_lrs[idt] =  180. - (slope_mcw[idt] + slope_fsa[idt])
            
        except:
            slope_fsa[idt] = np.nan
            slope_urs[idt] = np.nan
            slope_lrs[idt] = np.nan
        
        '''
        **************************************************************************
        '''
    
        #average rim height
        h[idt] = zi[idxF]
        
        # calculate the depth (new way where the min along each cross section is taken)
        depth_tmp = np.min(zi)
        depth[idt] = depth_tmp
        
        '''
        **************************************************************************
        '''
        
        # flank rim decay length
        interval_frdl = zi[idxF:idxH] # I think I can not use idxH+1 otherwise it is outside
        dist_frdl = dist[idxF:idxH]
        
        try:
            a, b = curve_fit(linear, dist_frdl, interval_frdl)
            xs = np.linspace(np.min(dist_frdl),np.max(dist_frdl),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_frdl = np.abs(tetarad * (180./np.pi))
            
            # get the maximum slope
            slope_frdl_max = np.nanmax(slope_frdl)
            
            # where it is the closest of half the maximum
            __, idx_frdl = find_nearest(slope_frdl, slope_frdl_max/2.0)
            
            # get the distance at half the maximum
            frdl[idt] = dist_frdl[idx_frdl]
            
        except:
            frdl[idt] = np.nan
            
        '''
        **************************************************************************
        '''
            
        # cavity rim decay length
        interval_crdl = zi[idxA:idxF+1] # I think I can not use idxH+1 otherwise it is outside
        dist_crdl = dist[idxA:idxF+1]
        
        try:
            a, b = curve_fit(linear, dist_crdl, interval_crdl)
            xs = np.linspace(np.min(dist_crdl),np.max(dist_crdl),100)
            ys = linear(xs,*a)
            
            #calculate the slope
            tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
            slope_crdl = np.abs(tetarad * (180./np.pi))
            
            # get the maximum slope
            slope_crdl_max = np.nanmax(slope_crdl)
            
            # where it is the closest of half the maximum
            __, idx_crdl = find_nearest(slope_crdl, slope_crdl_max/2.0)
            
            # get the distance at half the maximum
            crdl[idt] = dist_frdl[idx_crdl]
            
            # volume gets calculate afterwards from calculate_volume.py
            
        except:
            crdl[idt] = np.nan
        
        idt = idt + 1
    
        '''
        **************************************************************************
        '''
        
        
    return (R_upcw, R_ufrc, cse, slope_mcw, slope_ucw, slope_fsa, slope_lrs, slope_urs, crdl, frdl,
            h, depth, diamd, len(idx_circle2), prof_uni_detected, crossSections, YSections, XSections)



   
'''
******************************************************************************
'''

import re


def tokenize(filename):
    '''
    Function to list filenames in correct order (see
    http://stackoverflow.com/questions/5997006/sort-a-list-of-files-using-python)

    :param filename:
    :return:
    '''
    digits = re.compile(r'(\d+)')

    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))
    
    
'''
******************************************************************************
'''
