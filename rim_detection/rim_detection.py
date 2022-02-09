from rasterio.plot import reshape_as_raster, reshape_as_image


import copy
import numpy as np
import random
import rasterio as rio
import scipy.linalg
import scipy.ndimage

  
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
    n_points_along_cs = np.int((crater_radius / dem_resolution) * 2.0)
    
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

def maximum_elevation(z, n_points_along_cs, heights, widths, height_mesh, width_mesh):
    '''
    Find the maximum elevation along a cross section (1-D numpy array).

    Parameters
    ----------
    z : 1-D numpy array
        Elevations along a cross-section.
    n_points_along_cs : int
        Number of sampled points along the cross-section.
    heights : int
        all positions (in terms of indexes in the array) across the x-axis.
    widths : int
        all positions (in terms of indexes in the array) across the y-axis.
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    height_coord_ME : numpy array
        position of the maximum_elevation in DEM map projection accross the x-axis.
    width_coord_ME : numpy array
        position of the maximum_elevation in DEM map projection accross the y-axis.
    height_px_ME : numpy array
        position of the maximum_elevation in px coordinates accross the x-axis.
    width_px_ME : numpy array
        position of the maximum_elevation in px coordinates accross the y-axis.
    maximum_elevation : float
        maximum elevation along the cross-section profile.

    '''
    
    # I still don't understand how nan values will occur
    max_elv_idx = np.nanargmax(z)
        
    # we do not want to take the two last values (as it will likely not represent
    # the rim but the ridge/ambient topography)
    if max_elv_idx >= n_points_along_cs - 2: 
        height_coord_ME = np.nan
        width_coord_ME = np.nan
        
        maximum_elevation = np.nan
        
        height_px_ME = np.nan
        width_px_ME = np.nan
        
    # we do not want to take the values at the centre of the crater 
    # if for some reasons it should detect the higher elevation there
    # (I run into an example where it was like that)
    elif max_elv_idx == 0:
        height_coord_ME = np.nan
        width_coord_ME = np.nan
        
        maximum_elevation = np.nan
        
        height_px_ME = np.nan
        width_px_ME = np.nan
        
    # else we just take the x- and y-coordinates and the elevation of the 
    # maximum elevations
    else:        
        maximum_elevation = z[max_elv_idx]
        
        height_px_ME = int(heights[max_elv_idx])
        width_px_ME= int(widths[max_elv_idx])
        
        height_coord_ME = height_mesh[height_px_ME,width_px_ME]
        width_coord_ME = width_mesh[height_px_ME,width_px_ME]      
        
    return (height_coord_ME, width_coord_ME, 
            height_px_ME, width_px_ME, 
            maximum_elevation)

# changed name from local_elevation to local_elevations
# search_ncells is added so that not too many local minimas are detected
def local_elevations(z, n_points_along_cs, search_ncells, 
                     heights, widths, 
                     cs_id, 
                     height_mesh, width_mesh):
    '''
    Parameters
    ----------
    z : numpy 1-D array
        Elevations along the cross-section profile.
    n_points_along_cs : int
        Number of elevations along the cross section profile.
    search_ncells : int
        Number of cells to search before and after a local minima is found.
    heights : numpy 1-D array (int)
        x-coordinate of all the elevations in the cross-section (in pixel).
    widths : numpy 1-D array (int)
        y-coordinate of all the elevations in the cross-section (in pixel).
    cs_id : int
        cross-section id.
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    height_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the x-axis).
    width_coord_LE : numpy 1-D array (float)
        positions of the local elevations in the DEM in map projection (across the y-axis).
    height_px_LE : numpy 1-D array (int)
        positions of the local elevations in the DEM in pixels (across the x-axis).
    width_px_LE : numpy 1-D array (int)
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
    (height_coord_LE, width_coord_LE, 
     height_px_LE, width_px_LE, 
     elev_LE, prof_LE) = [np.array([]) for _ in range(6)] 

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
                c = int(heights[i])
                r = int(widths[i])
                
                height_coord_LE = np.append(height_coord_LE, height_mesh[c,r]) 
                width_coord_LE = np.append(width_coord_LE, width_mesh[c,r])
                height_px_LE = np.append(height_px_LE, c)
                width_px_LE = np.append(width_px_LE, r)                                        
                elev_LE = np.append(elev_LE, z[i])
                prof_LE = np.append(prof_LE, cs_id)
                
    return (height_coord_LE, width_coord_LE, 
            height_px_LE, width_px_LE, 
            elev_LE, prof_LE)

def slope_change(z, distances_along_cs, search_ncells, minslope_diff, 
                 clustersz, heights, widths, height_mesh, width_mesh):
    '''
    This function detects where the biggest change in slopes happen along a 
    cross section and returns both map and pixel coordinates and the corresponding
    elevation.

    Parameters
    ----------
    z : numpy 1-D array
        Elevations along the cross-section profile.
    distances_along_cs : numpy 1-D array
        Distances along the cross section profile.
    search_ncells : int
        Number of cells to search before and after a location along the cs
    minslope_diff : float
        Minimum value for a difference in slope to be flag.
    clustersz : int
        Minimum value for a cluster of slope values.
    heights : numpy 1-D array (int)
        x-coordinate of all the elevations in the cross-section (in pixel).
    widths : numpy 1-D array (int)
        y-coordinate of all the elevations in the cross-section (in pixel).
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.

    Returns
    -------
    height_coord_BS : numpy 1-D array (float)
        positions of the maximum change in slopes in the DEM in map projection 
        (across the x-axis).
    width_coord_BS : numpy 1-D array (float)
        positions of the maximum change in slopes in the DEM in map projection 
        (across the y-axis).
    height_px_BS : numpy 1-D array (int)
        positions of the maximum change in slopes in the DEM in pixels 
        (across the x-axis).
    width_px_BS : numpy 1-D array (int)
        positions of the maximum change in slopes in the DEM in pixels 
        (across the y-axis).
    elevation_change_in_slope : numpy 1-D array (float)
        elevation at the maximum change in slopes
    '''    
    # create empty arrays    
    (slopebe4, slopeaft) = [np.zeros(len(z)) for _ in range(2)]
    
    # we need to test if these are the largest values within 0.1*R of the value
    for i in np.arange(search_ncells,len(distances_along_cs)-search_ncells,1):
        
        # find s and z value sto construct a slope
        sbe4 = np.array([distances_along_cs[(i-search_ncells):i], 
                         np.ones(search_ncells)]).T
        
        zbe4 = z[(i-search_ncells):i]
        
        saft = np.array([distances_along_cs[i:(i+search_ncells)], 
                         np.ones(search_ncells)]).T
        
        zaft = z[i:(i+search_ncells)]
        
        # calculate slope 0.1R before and after the actual point along the cs
        be4stat = np.linalg.lstsq(sbe4,zbe4)[0]
        aftstat = np.linalg.lstsq(saft,zaft)[0]
            
        slopebe4[i] = be4stat[0]
        slopeaft[i] = aftstat[0]
            
    # substract to find max slope differences
    slope_diff = (slopebe4 - slopeaft)
    
    #make empty variables
    cluster = []
    ind = []
    
    # if slope_diff value is lower than the minslope_diff add to cluster
    for val in slope_diff:
        if val > minslope_diff:
            cluster.append(val)
            
        # once value goes below 0, cluster ends
        if val <= 0:
            if len(cluster) > clustersz:
                cluster = np.array(cluster)
                
                # automatically updating the pixel along the cs where the 
                # maximum change in slope is detected
                ind = (np.nonzero(slope_diff == np.max(cluster))[0][0])
            
            # cluster reset to empty list                       
            cluster = []
        
    if ind:
        c = int(heights[ind])
        r = int(widths[ind])
        height_coord_BS = height_mesh[c,r]
        width_coord_BS = width_mesh[c,r]
        height_px_BS = c
        width_px_BS = r
        elevation_change_in_slope = z[ind]
        
    else:
        height_coord_BS = np.nan
        width_coord_BS = np.nan
        height_px_BS = np.nan
        width_px_BS = np.nan
        elevation_change_in_slope = np.nan     
        
    return (height_coord_BS, width_coord_BS, 
            height_px_BS, width_px_BS, 
            elevation_change_in_slope)

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


def detect_maximum_elevations(height_mesh, width_mesh, 
                              height_crater_center_px, width_crater_center_px, 
                              elevations, crater_radius, dem_resolution, 
                              debugging=False):
    '''
    

    Parameters
    ----------
    height_mesh : TYPE
        DESCRIPTION.
    width_mesh : TYPE
        DESCRIPTION.
    height_crater_center_px : TYPE
        DESCRIPTION.
    width_crater_center_px : TYPE
        DESCRIPTION.
    elevations : TYPE
        DESCRIPTION.
    crater_radius : TYPE
        DESCRIPTION.
    dem_resolution : TYPE
        DESCRIPTION.

    Returns
    -------
    height_coord_ME : TYPE
        DESCRIPTION.
    width_coord_ME : TYPE
        DESCRIPTION.
    height_px_ME : TYPE
        DESCRIPTION.
    width_px_ME : TYPE
        DESCRIPTION.
    elev_ME : TYPE
        DESCRIPTION.
    prof_ME : TYPE
        DESCRIPTION.

    '''
    
   
    # 2R circles pixel coordinates height_circle_px, width_circle_px
    height_circle_px, width_circle_px = xy_circle((2.0*crater_radius) / dem_resolution, 
                         height_crater_center_px, width_crater_center_px)
    
    # real values height_circle_px, width_circle_px
    (height_circle_px, width_circle_px) = (np.round(height_circle_px).astype('int'), np.round(width_circle_px).astype('int'))
                    
    # we define the maximum elevation variables   
    (height_coord_ME, width_coord_ME, 
     height_px_ME, width_px_ME, 
     elev_ME) = [np.ones(len(height_circle_px)) for _ in range(5)]
    
    # profile ID
    prof_ME = np.arange(0,512)
    
    # samples at half the dem_resolution 
    n_points_along_cs = np.int(np.ceil(2.0*crater_radius/dem_resolution)*2.0)
    
    # set arrays equal to nan
    [array.fill(np.nan) for array in 
    [height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME]]
        
    # generate cross sections between the centre of the crater and the 2.0R
    # circle pixel coordinates
    for ix in range(len(height_circle_px)):

        # find the pixel coordinates 
        ncol = height_circle_px[ix]
        nrow = width_circle_px[ix]
                
        # the distance is calculated, should be equal to two times the crater_radius
        (heights, widths) = (np.linspace(height_crater_center_px, ncol, n_points_along_cs),
        np.linspace(width_crater_center_px, nrow, n_points_along_cs))
        
        # Extract the values along the line, using cubic interpolation and the 
        # map coordinates
        z = scipy.ndimage.map_coordinates(elevations, np.vstack((heights,widths)))
                
        # Maximum elevation     
        (height_coord_ME[ix], width_coord_ME[ix], height_px_ME[ix], width_px_ME[ix], 
         elev_ME[ix]) = maximum_elevation(z, n_points_along_cs, heights, widths, height_mesh, width_mesh)
    

    # TO DO: fix the implementation of debugging    
    if debugging:
        heights_list = []
        widths_list = []
        z_list = []
        prof_list = []
        
        for ix in range(len(height_circle_px)):
    
            # find the pixel coordinates 
            ncol = height_circle_px[ix]
            nrow = width_circle_px[ix]
                    
            # the distance is calculated, should be equal to two times the crater_radius
            (heights, widths) = (np.linspace(height_crater_center_px, ncol, n_points_along_cs),
            np.linspace(width_crater_center_px, nrow, n_points_along_cs))
            
            # Extract the values along the line, using cubic interpolation and the 
            # map coordinates
            z = scipy.ndimage.map_coordinates(elevations, np.vstack((heights,widths)))
            
            heights_list.append(heights)
            widths_list.append(widths)
            z_list.append(z)
            prof_list.append([ix]*len(heights))
        
        
        return (heights_list, widths_list, z_list, prof_list,
                height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME, prof_ME)
    else:
        return (height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME, prof_ME)
    
    

def detect_maximum_and_local_elevations(height_mesh, width_mesh, 
                                        height_crater_center_px, width_crater_center_px, 
                                        elevations, crater_radius, dem_resolution):
    '''
    Routine to extract the maximum elevations, local minima elevations and
    max. break in slope elevations from a given crater. 

    Parameters
    ----------
    height_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    width_mesh : numpy array
        mesh grid with the same dimension as the elevations.
    height_crater_center_px : int
        centre of the crater in pixel coordinates.
    width_crater_center_px : int
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
    # Let's set some of the values (hard-coded)
    mingrade = 0.05
    minclust = 0.05
    slen = 0.1

     
    # 2R circles pixel coordinates height_circle_px, width_circle_px
    height_circle_px, width_circle_px = xy_circle((2.0*crater_radius) / dem_resolution, 
                         height_crater_center_px, width_crater_center_px)
    
    # real values height_circle_px, width_circle_px
    (height_circle_px, width_circle_px) = (np.round(height_circle_px).astype('int'), np.round(width_circle_px).astype('int'))
                  
    # we define the maximum elevation variables   
    (height_coord_ME, width_coord_ME, 
     height_px_ME, width_px_ME, 
     elev_ME) = [np.ones(len(height_circle_px)) for _ in range(5)]
    
    prof_ME = np.arange(0,512)
    
    # we define the local maxima variables
    (height_coord_LE, width_coord_LE, 
     height_px_LE, width_px_LE, 
     elev_LE, prof_LE) = [np.array([]) for _ in range(6)] 
    
    # we define the break in slope variables
    (height_coord_BS, width_coord_BS, 
     height_px_BS, width_px_BS, 
     elev_BS) = [np.ones(len(height_circle_px)) for _ in range(5)]
    
    prof_BS = np.arange(0,512)
    
    
    # set arrays equal to nan
    [array.fill(np.nan) for array in 
     [height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME]]
    
    [array.fill(np.nan) for array in 
     [height_coord_BS, width_coord_BS, height_px_BS, width_px_BS, elev_BS]]
    
    # samples at half the dem_resolution 
    n_points_along_cs = np.int(np.ceil(2.0*crater_radius/dem_resolution)*2.0)
    
    # calculate the threshold (which is 0.1 times the scaled radius) 
    # in terms of distance and number of cells
    thres = 0.1 * crater_radius
    ncells = np.int(np.ceil(thres/dem_resolution))
    
    # find out how many points will be in each slope segment
    slopenum = int(n_points_along_cs*slen) #slen = 0.1
    clustersz = n_points_along_cs*minclust #minclust = 0.05 is it used some where?
    
    # generate cross sections between the centre of the crater and the 2.0R
    # circle pixel coordinates
    for ix in range(len(height_circle_px)):

        # find the pixel coordinates 
        ncol = height_circle_px[ix]
        nrow = width_circle_px[ix]
                
        # the distance is calculated, should be equal to two times the crater_radius
        (heights, widths) = (np.linspace(height_crater_center_px, ncol, n_points_along_cs),
        np.linspace(width_crater_center_px, nrow, n_points_along_cs))
        
        # Extract the values along the line, using cubic interpolation and the 
        # map coordinates
        z = scipy.ndimage.map_coordinates(elevations, np.vstack((heights,widths)))
        
        # calculate the distances along the profile
        dist_px = np.sqrt(((heights - height_crater_center_px)**2.) + ((widths - width_crater_center_px)**2.))
        dist = dist_px * dem_resolution 
        
        # Detect the maximum elevation along each cross-section     
        (height_coord_ME[ix], width_coord_ME[ix], height_px_ME[ix], width_px_ME[ix], 
         elev_ME[ix]) = maximum_elevation(z, n_points_along_cs, heights, widths, height_mesh, width_mesh)
              
        # Detect local elevations along each cross-section
        (height_coord_LE_tmp, width_coord_LE_tmp, height_px_LE_tmp, 
         width_px_LE_tmp, elev_LE_tmp, prof_LE_tmp) = local_elevations(z, 
                                                   n_points_along_cs, ncells, heights, widths, ix, 
                                                   height_mesh, width_mesh)
                                                                     
        # can be several local elevations per cross sections
        height_coord_LE = np.append(height_coord_LE, height_coord_LE_tmp)
        width_coord_LE = np.append(width_coord_LE, width_coord_LE_tmp)
        height_px_LE = np.append(height_px_LE, height_px_LE_tmp)
        width_px_LE = np.append(width_px_LE, width_px_LE_tmp)
        elev_LE = np.append(elev_LE, elev_LE_tmp)
        prof_LE = np.append(prof_LE, prof_LE_tmp)
        
        # break in slopes
        (height_coord_BS[ix], width_coord_BS[ix], height_px_BS[ix], 
         width_px_BS[ix], elev_BS[ix]) = (slope_change(z, dist, slopenum, 
                              mingrade, clustersz, heights, widths, height_mesh, width_mesh))

            
    return (height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME, prof_ME,
            height_coord_LE, width_coord_LE, height_px_LE, width_px_LE, elev_LE, prof_LE,
            height_coord_BS, width_coord_BS, height_px_BS, width_px_BS, elev_BS, prof_BS)





def rim_composite(height_coord_ME, width_coord_ME, height_px_ME, width_px_ME, elev_ME, profile_ME, 
                  height_coord_LE, width_coord_LE, height_px_LE, width_px_LE, elev_LE, profile_LE,
                  height_mesh, width_mesh, height_crater_center_px, width_crater_center_px, 
                  maximum_shift_ME, maximum_shift_LE):
    '''
    

    Parameters
    ----------
    height_coord_ME : TYPE
        DESCRIPTION.
    width_coord_ME : TYPE
        DESCRIPTION.
    height_px_ME : TYPE
        DESCRIPTION.
    width_px_ME : TYPE
        DESCRIPTION.
    elev_ME : TYPE
        DESCRIPTION.
    profile_ME : TYPE
        DESCRIPTION.
    height_coord_LE : TYPE
        DESCRIPTION.
    width_coord_LE : TYPE
        DESCRIPTION.
    height_px_LE : TYPE
        DESCRIPTION.
    width_px_LE : TYPE
        DESCRIPTION.
    elev_LE : TYPE
        DESCRIPTION.
    profile_LE : TYPE
        DESCRIPTION.
    height_mesh : TYPE
        DESCRIPTION.
    width_mesh : TYPE
        DESCRIPTION.
    height_crater_center_px : TYPE
        DESCRIPTION.
    width_crater_center_px : TYPE
        DESCRIPTION.
    maximum_shift_ME : TYPE
        DESCRIPTION.
    maximum_shift_LE : TYPE
        DESCRIPTION.

    Raises
    ------
    an
        DESCRIPTION.

    Returns
    -------
    candidates_rim_composite : TYPE
        DESCRIPTION.
    n_ME_not_used : TYPE
        DESCRIPTION.
    gaplist : TYPE
        DESCRIPTION.
    delta_distances : TYPE
        DESCRIPTION.

    '''
    # Let's hard code some of the values (we don't change them actually)
    angle = 2.0 #(in degrees)                
    stangle = [0,45,90,135,180,225,270,315]                
    contloop = True
    siftRedundant = True
    
    # number of maximum elevation detections (should be 512)
    nME_detection = len(height_coord_ME)
    
    # converted from degrees to our circle divided in 512 radial profiles
    # we will have a cross section every three degrees
    angle = np.ceil(angle* (nME_detection/360.))
    
    # equivalent points of 0, 45, 90, 135 degrees in terms of our data
    starting_crossS_id = (nME_detection * np.array(stangle))/(360.)
    starting_crossS_id = starting_crossS_id.astype('int')
    
    # defining empty variables
    distance_to_ME_detection = np.zeros(nME_detection)
    height_width_coord_ME_detection = np.zeros((2,nME_detection))
    height_width_coord_LE_detection = np.zeros((2,len(height_coord_LE)))
    candidates_rim_composite = []
    n_ME_not_used = []
    delta_distances = []
    gaplist = []
    
    '''
    ***********************MAXIMUM ELEVATION**********************************
    '''
    #Only work with not nan values (previous nan-values will be equal to 0)
    nnan = np.where(np.isfinite(height_px_ME))
    
    # calculate the crater_radius to the global maximum elevation
    ab = (width_coord_ME[nnan] - width_mesh[height_crater_center_px,width_crater_center_px])**2.0 #changed 
    bc = (height_coord_ME[nnan] - height_mesh[height_crater_center_px,width_crater_center_px])**2.0 #changed
    distance_to_ME_detection[nnan] = np.sqrt(ab + bc)
    
    # get the indices of the global maximum elevation
    height_width_coord_ME_detection[0,[nnan]] = height_px_ME[nnan]
    height_width_coord_ME_detection[1,[nnan]] = width_px_ME[nnan]
    height_width_coord_ME_detection = height_width_coord_ME_detection.astype('int')
    
    '''
    ***********************LOCAL ELEVATION************************************
    '''
    
    # calculate the distances to the local maximum elevations
    ab = (width_coord_LE - width_mesh[height_crater_center_px,width_crater_center_px])**2.0 #changed
    bc = (height_coord_LE - height_mesh[height_crater_center_px,width_crater_center_px])**2.0 #changed
    distances_to_LE_detection = np.sqrt(ab + bc)
    
    # get the indices of the local maximum elevation
    height_width_coord_LE_detection[0,:] = height_px_LE
    height_width_coord_LE_detection[1,:] = width_px_LE
    height_width_coord_LE_detection =height_width_coord_LE_detection.astype('int')
    
    '''
    ***********************LOOPS *********************************************
    '''
    
    for strt in starting_crossS_id:
                
        #counter clockwise loop
        ccw = np.concatenate((np.arange(strt,nME_detection),np.arange(0,strt)))
        
        # clockwise loop
        cw = np.concatenate(((np.arange(strt+1)[::-1]),np.arange(strt+1,nME_detection)[::-1]))
        
        # take both loops
        loops = [cw, ccw]
        
        # count the number of counter clockwise and clockwise loops
        # for example for a stangle = [0, 90, 180, 270], we should have 4 
        # starting points looping clockwise and counterclockwise (2), so we will
        # have 8 canidates rim composite       
        pnum = 0 
        
        for crossS_id in loops:
                      
            #create empty rim trace for this path
            #This variable will contain the height, width, elevation, crossSID
            # and the type of elevations added (ME or LE)
            candidate_rim_composite = np.zeros([5,nME_detection]) 
            
            #differences between distances to LE/ME (in between two consecutive
            # cross sections ids) 
            delta_distance = [] 
            
            # Find last point of loop to have reference for start
            # We need to find the maximum elevation for this profile
            distance_to_last_rim_composite_elevation = distance_to_ME_detection[crossS_id[-1]]
            
            # in case there are no maximum elevation in this profile           
            # search in the profile before until it find a value
            previous_id = -2
            
            # continue until it is not equal to zero anymore
            while (distance_to_last_rim_composite_elevation == 0):
                distance_to_last_rim_composite_elevation = distance_to_ME_detection[crossS_id[previous_id]]
                previous_id -=1
                if np.abs(previous_id) >= nME_detection:
                    break
                    
                #I had an error where no distance_to_last_rim_composite_elevation have been found
                #it loops until -513 (which is out of bounds)
                
            # define some variables to start with
                
            # number of times when the ME point failed to be within 0.1R of the 
            # previous ME distance
            n_ME_failed = 0 
            k = 0
            gap = 0
            before = False
            
            #
            while before == False:
                
                # what is the index (i.e., the cross-section profile id)
                i = crossS_id[k]
                
                # idx_LE_candidates and idx_rim_candidate are resetted at the start
                
                # indexes to local Elevation candidates along the specific cross section
                idx_LE_candidates = []
                
                # index to the final rim elevation candidate (index either of the ME or LE)  
                idx_rim_candidate = []        
                
                # is the crater_radius of the global maximum at that cs id within 0.10R
                # of the previous estimate?
                ub = np.ceil(distance_to_last_rim_composite_elevation + maximum_shift_ME) # upper-boundary
                lb = np.floor(distance_to_last_rim_composite_elevation - maximum_shift_ME) # lower-boundary
                
                # if yes, save the csid profile
                if np.logical_and(distance_to_ME_detection[i] <= ub, distance_to_ME_detection[i] >= lb):
                    
                    idx_rim_candidate = i
                    
                    # flag == 0 ---> Maximum elevation is used
                    # flag == 1 ---> Local elevation is used
                    # flag == 2 ---> Neither maximum or local elevations are used
                    flag = 0
                    
                #if not look for other interest points
                else:
                    
                    # keep track of # off the max method
                    n_ME_failed += 1
                    
                    # search in local maxima, are they some candidates?
                    posI = np.where(profile_LE == i)
                    
                    # if not (should actually not go through this loop so often)
                    if len(posI[0]) == 0:
                                                                                
                        # assign gap and move
                        gap += 1
                        k += 1
                        
                        # if we are at the end start over
                        if k >= len(crossS_id):
                            k = 0
                        continue
                    
                    
                    # not sure about the place of this thing
                    if maximum_shift_LE:
                        
                        # reset angular distance
                        adis = 0
                        ak = k
                        aind = crossS_id[ak]
                        #sti = []
                        
                        while adis <= angle:
                        
                            #find all possible candidates in next spoke
                            posI = np.where(profile_LE == aind)[0]
                            
                            #get their crater_radius
                            posR = distances_to_LE_detection[posI] - distance_to_last_rim_composite_elevation
                            
                            #find the radial distance of the candidate from laspnt
                            idx_LE_candidates = posI[np.nonzero(np.abs(posR) <= maximum_shift_LE)]
                            
                            # if candidates is empty random.py will raise an error
                            # so we have to use try
                            
                            try:
                                # take a random local elevations within 0.05D
                                # of the initially estimated crater diameter. 
                                idx_rim_candidate = random.choice(idx_LE_candidates)
                                flag = 1
                                
                                # then stop the while loop
                                break
                            #if no local elevations within 0.05D, go to the next profile
                            except:
                                adis += 1
                                ak += 1
                                if ak == len(crossS_id):
                                    ak = 0
                                aind = crossS_id[ak]
                                
                            # after the random candidate local elevations is selected
                            # check if it is within 3 cross sectional profiles
                            if ((adis > 0) & (adis <= angle)): #if something was skipped
                                                               
                                i = aind
                                k = ak
                                
                # if it does not work just pick the closest (not within 0.05D)        
                if idx_rim_candidate == []:
                    posI = np.where(profile_LE == i)[0]
                                                
                    #get their crater_radius
                    posR = np.abs(distances_to_LE_detection[posI] - distance_to_last_rim_composite_elevation)
                    
                    #I guess some of the largest radial discontinuities should
                    # happen here as others should be within 0.05*r or 0.1*r
                    # in case there are no data
                    if len(posR >= 1):
                        delta_distance.append(np.min(posR))
                        
                        #candidate is closest
                        idx_LE_candidates = posI[np.nanargmin(posR)]
                        
                        # if several with the same minimal distance # don't need 
                        # with nan argmin
                        idx_rim_candidate = idx_LE_candidates
                        flag = 1
                    else:
                        flag = 2
                    
                # special for me
                if flag == 1:
                    height_tmp = height_coord_LE[idx_rim_candidate]
                    width_tmp = width_coord_LE[idx_rim_candidate]
                    elevation_tmp = elev_LE[idx_rim_candidate]
                    
                    # Assign new last R
                    distance_to_last_rim_composite_elevation = distances_to_LE_detection[idx_rim_candidate]
                    
                elif flag == 0:
                    height_tmp = height_coord_ME[idx_rim_candidate]
                    width_tmp = width_coord_ME[idx_rim_candidate]
                    elevation_tmp = elev_ME[idx_rim_candidate]
                    
                    # Assign new last R
                    distance_to_last_rim_composite_elevation = distance_to_ME_detection[idx_rim_candidate]
                    
                else:
                    #if no values are found at all (flag == 2)
                    height_tmp = np.nan
                    width_tmp = np.nan
                    elevation_tmp = np.nan
                    
                    
                if candidate_rim_composite[0,i] == height_tmp:
                    if candidate_rim_composite[1,i] == width_tmp:
                        if candidate_rim_composite[2,i] == elevation_tmp:
                            before = True                                    
                
                # change height_tmp to height_tmp
                # change width_tmp to width_tmp
                candidate_rim_composite[0,i] = height_tmp
                candidate_rim_composite[1,i] = width_tmp
                candidate_rim_composite[2,i] = elevation_tmp
                candidate_rim_composite[3,i] = crossS_id[k] # new it takes where in the profile
                candidate_rim_composite[4,i] = flag

                
                # increment the path loop
                k += 1
                
                # if at the end of the path start over
                if k >= len(crossS_id):
                    k = 0
                    if contloop == False:
                        before = True
                                
                                
            # when finished, divide the points off Maxmethod by total
            n_ME_failed = n_ME_failed / float(nME_detection) 
            aprim = True
    
            # if not the first, check if (don't really understand this part)
            
            
            # if length of the rim composite is larger than zero?
            if len(candidates_rim_composite) > 0:
                
                # loop through candidates_rim_composite (and then do something I don't understand)
                # I guess is that if there are exactely the same then it just skip to 
                # save the data. This code of lines does not do too much I could just skip
                # it..... I guess it's speeding things up as 16 cross sectional values could be 
                # a lot of data sometimes
                
                for j in range(len(candidates_rim_composite)):
                    if sum(candidate_rim_composite[0,:] != candidates_rim_composite[j][0,:]) == 0:
                        #print "Trace is same as Trace " + str(j)
                        if siftRedundant:
                            aprim = False
                        break
            
            # can not create a numpy array as the number of values per cross section 
            # can change?
            if aprim:
                
                # make a copy of the x, y, z, crossS_id, flag (0 or 1, ME or LE)
                candidates_rim_composite.append(copy.deepcopy(candidate_rim_composite))
                
                # index when the ME point failed to be within 0.1R of the 
                # previous ME distance
                n_ME_not_used.append(n_ME_failed) 
                
                #number of gap (if absolutelty nothing worked)
                gaplist.append(gap) 
                
                # all the difference between distances to LE and previously 
                # calculated distance to rim
                delta_distance = np.array(delta_distance) 
                delta_distance_sorted = delta_distance[np.argsort(delta_distance)][-5:] # take the five biggest discrepancies?
                delta_distances.append(delta_distance_sorted)
            
            # to continue to loop through all possible clockwise, counterclockwise
            pnum +=1
    
    return (candidates_rim_composite, n_ME_not_used, gaplist, delta_distances)
    
    
    
def run(crater_dem, crater_radius, scaling_factor):
    
    '''
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
    # read array
    with rio.open(crater_dem) as src:
        array = reshape_as_image(src.read())[:,:,0]
        meta = src.profile
    
    # infer dem resolution from the crater dem 
    dem_resolution = meta['transform'][0]
    
    # height scaling factor for the SLDEM2015
    z = array * scaling_factor
    
    # we would like to have a square array so we take the min
    min_of_shape = np.min(z.shape)
    z_square = z[:min_of_shape, :min_of_shape]

    x_y = np.linspace(0,(min_of_shape-1)*dem_resolution,min_of_shape)
    x_y_center = np.linspace(dem_resolution/2.0, (dem_resolution/2.0) + 
                             ((min_of_shape-1)*dem_resolution), min_of_shape)
    
    height_mesh, width_mesh = np.meshgrid(x_y, x_y, indexing='ij') # new addition
    height_mesh_center, width_mesh_center = np.meshgrid(x_y_center, x_y_center, indexing='ij')
    
    # centre of the map
    x_center_px = int(min_of_shape / 2)
    y_center_px = int(min_of_shape / 2)
    
    
    filterMedianStd = True #use standard deviation removal
    
    # Only a single detrending is first run
    z_detrended = detrending(crater_radius,
                   2.0, 3.0,
                   z_square, 
                   dem_resolution, 
                   height_mesh, width_mesh, 
                   x_center_px, y_center_px, 
                   filterMedianStd)
       
    # Return maximum, local and change in slope elevations are detected
    return (z_detrended, detect_maximum_and_local_elevations(height_mesh, width_mesh, x_center_px, 
                                             y_center_px, z_detrended, 
                                             crater_radius, dem_resolution))
    
    
