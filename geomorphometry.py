import numpy as np
import scipy
import scipy.ndimage
from scipy.optimize import curve_fit
from scipy import optimize

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

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def calc_D(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    
    xt = x[~np.isnan(x)]
    yt = y[~np.isnan(y)]
    
    x_m = np.nanmean(xt)
    y_m = np.nanmean(yt)
    center_estimate = x_m, y_m
    
    center, ier = optimize.leastsq(calc_D, center_estimate, args=(xt,yt))    
    xc, yc = center
    Ri = calc_R(xt, yt, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def power(x,a,b,c):   
    return a * (x**b) + c

def linear(x,a,b):
    return a*x + b


def geomorphometry_cs(z, dist, dist_norm):
    
    # Normalized distances along a cross section
    A , idxA = find_nearest(dist_norm,0.0)
    B , idxB = find_nearest(dist_norm,0.1)
    C , idxC = find_nearest(dist_norm,0.7)
    D , idxD = find_nearest(dist_norm,0.8)
    E , idxE = find_nearest(dist_norm,0.9)
    F , idxF = find_nearest(dist_norm,1.0)
    G , idxG = find_nearest(dist_norm,1.2)
    H , idxH = find_nearest(dist_norm,2.0)
    
    # upper cavity wall roc
    ucw_roc_val = radius_of_curvature(z[idxD:idxF+1], dist[idxD:idxF+1])
    
    # upper flank roc
    uf_roc_val = radius_of_curvature(z[idxF:idxG+1], dist[idxF:idxG+1])
    
    # cavity shape exponent
    cse_val = cavity_shape_exponent(z[idxB:idxE+1], dist[idxB:idxE+1])
    
    # middle cavity slope  
    slope_mcw = cavity_slope(z[idxC:idxE+1], dist[idxC:idxE+1])
    
    # upper cavity slope
    slope_ucw = cavity_slope(z[idxD:idxF+1], dist[idxD:idxF+1])
    
    # flank slope angle
    slope_flank = np.abs(cavity_slope(z[idxF:idxG+1], dist[idxF:idxG+1]))
    
    # upper rim span
    slope_urs = 180. -  (slope_ucw + slope_flank)
    
    # lower rim span
    slope_lrs =  180. - (slope_mcw + slope_flank)
    
    # average rim height
    h = z[idxF]
    
    # rim height from the minimum elevation beyond the rim (up to a distance of
    # 2R)
    hr = rim_height_above_minimum_height_beyond_rim(z[idxF], z[idxF:idxH])
    
    # calculate the depth (new way where the min along each cross section is taken)
    depth = np.min(z)
    
    # flank rim decay length
    frdl = rim_decay_length(z[idxF:idxH], dist[idxF:idxH], dist[idxF])
    
    # cavity rim decay length
    crdl = rim_decay_length(z[idxA:idxF+1], dist[idxA:idxF+1], dist[idxF])
    
    
    return (depth, h, hr, cse_val, slope_mcw, slope_ucw, slope_flank, slope_urs,
            slope_lrs, frdl, crdl, ucw_roc_val, uf_roc_val)

    
    

def radius_of_curvature(elevations_roc, distances_roc):
    '''
    

    Parameters
    ----------
    elevations_roc : TYPE
        DESCRIPTION.
    distances_roc : TYPE
        DESCRIPTION.

    Returns
    -------
    roc_val : TYPE
        DESCRIPTION.

    '''
    
    try:
        __, __, roc_val, __ = leastsq_circle(distances_roc, elevations_roc)
            
    except:
        roc_val = np.nan
        
    return (roc_val)

    
def rim_decay_length(elevations_rdl, distances_rdl, distance_rim):
    '''
    

    Parameters
    ----------
    elevations_rdl : TYPE
        DESCRIPTION.
    distances_rdl : TYPE
        DESCRIPTION.
    distance_rim : TYPE
        DESCRIPTION.

    Returns
    -------
    rdl : TYPE
        DESCRIPTION.

    '''
    
    try:
        x1 = distances_rdl[:-1]
        x2 = distances_rdl[1:]
        y1 = elevations_rdl[:-1]
        y2 = elevations_rdl[1:]
        
        dx = x2 - x1
        dy = y2 - y1
        
        tetarad = np.arctan(dy/dx)
        slope = np.abs(tetarad * (180./np.pi))
        
        # get the maximum slope
        slope_max = np.nanmax(slope)
        
        # where it is the closest of half the maximum
        __, idx_frdl = find_nearest(slope, slope_max/2.0)
        
        # get the distance at half the maximum
        rdl = np.abs(distances_rdl[idx_frdl] - distance_rim)
        
    except:
        rdl = np.nan
        
    return rdl
    
def cavity_shape_exponent(elevations_cse, distances_cse):
    '''
    

    Parameters
    ----------
    elevations_cse : TYPE
        DESCRIPTION.
    distances_cse : TYPE
        DESCRIPTION.

    Returns
    -------
    cse : TYPE
        DESCRIPTION.

    '''
    try:
        a, b = curve_fit(power, distances_cse, elevations_cse)
            
        exponent = a[1]
        cse = exponent
        
    except:
        cse = np.nan
        
    return (cse)

def cavity_slope(elevations_cavity, distances_cavity):
    '''
    

    Parameters
    ----------
    elevations_cavity : TYPE
        DESCRIPTION.
    distances_cavity : TYPE
        DESCRIPTION.

    Returns
    -------
    slope : TYPE
        DESCRIPTION.

    '''
       
    try:
        a, b = curve_fit(linear, distances_cavity, elevations_cavity)
        xs = np.linspace(np.min(distances_cavity),np.max(distances_cavity),100)
        ys = linear(xs,*a)
        
        #calculate the slope
        tetarad = np.arctan((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
        slope = tetarad * (180./np.pi)
        
    except:
        slope = np.nan
        
    return (slope)
    


def rim_height_above_minimum_height_beyond_rim(elevation_hr, 
                                               elevations_beyond_rim):
    '''
    Parameters
    ----------
    elevation_hr : TYPE
        DESCRIPTION.
    elevations_beyond_rim : TYPE
        DESCRIPTION.

    Returns
    -------
    hr : TYPE
        DESCRIPTION.

    '''

    try:
        min_h = np.nanmin(elevations_beyond_rim)
        
        # height from the rim to the smallest elevation beyond the rim 
        hr = elevation_hr - min_h
        
    except:
        hr = np.nan
        
    return (hr)


def find_unique(y_height_final, x_width_final, dem_resolution, cs):
    '''
    

    Parameters
    ----------
    x_rim : TYPE
        DESCRIPTION.
    y_rim : TYPE
        DESCRIPTION.
    dem_resolution : TYPE
        DESCRIPTION.
    cs : TYPE
        DESCRIPTION.

    Returns
    -------
    unique_i : TYPE
        DESCRIPTION.
    prof_uni_detected : TYPE
        DESCRIPTION.

    '''
    
    # create empty array
    idx_detected = np.zeros((len(y_height_final),2))
    idx_detected[:,0] = (y_height_final)/dem_resolution # transform to pixel coord
    idx_detected[:,1] = (x_width_final)/dem_resolution # transform to pixel coord
    idx_detected = idx_detected.astype('int')
    
    # Find unique indices from detected 
    __, unique_index = np.unique(["{}{}".format(i, j) for i,j in idx_detected], 
                          return_index=True)
    
    unique_i = idx_detected[unique_index,:] 
    prof_uni_detected = cs[unique_index] #still contain zeros
    
    return (unique_i, prof_uni_detected)


def calculate(y_height_final, x_width_final, profile_final,
              crater_radius,
              z, dem_resolution,
              y_height_center_px, x_width_center_px):

    # Find unique indices from detected rim
    # Note that the values are transfored to pixel coordinates in this step
    # (hidden in the function)
    index_unique, cs_unique = find_unique(y_height_final, x_width_final,
                                          dem_resolution, profile_final)

    # 2r from the center of the  crater (the origin is included here)
    idx_circle2 = np.zeros((len(index_unique), 2))
    idx_circle2[:, 0] = ((index_unique[:,
                          0] - y_height_center_px) * 2.) + y_height_center_px
    idx_circle2[:, 1] = ((index_unique[:,
                          1] - x_width_center_px) * 2.) + x_width_center_px

    # samples at half the dem_resolution
    num = np.int(np.ceil(2.0 * crater_radius / dem_resolution) * 2.0)

    # I need to define all empty arrays here
    (diameter, depth, h, hr, crdl, frdl, slope_urs, slope_lrs,
     slope_fsa, slope_ucw, slope_mcw, cse, uf_roc_val, ucw_roc_val) = [
        np.zeros(len(idx_circle2)) for _ in range(14)]

    # dictionary to save the cross sections to (can be save as a geopandas)
    crossSections = dict()
    XSections = dict()
    YSections = dict()

    # so it's only looping through cross sections where the rim composite
    # function resulted in a detection

    for i, ind in enumerate(idx_circle2):
        ncol = ind[0]
        nrow = ind[1]

        jj = index_unique[i]
        ncol_1r = jj[0]
        nrow_1r = jj[1]

        # the distance is calculated, should be equal to two times the crater_radius
        cols, rows = np.linspace(y_height_center_px, ncol, num), np.linspace(
            x_width_center_px, nrow, num)

        # Extract the values along the line, using cubic interpolation and the
        # map coordinates
        z_extracted = scipy.ndimage.map_coordinates(z, np.vstack(
            (cols, rows)))  # changed here

        # calculate the distance along the profile 2
        dist_px = np.sqrt(((cols - y_height_center_px) ** 2.) + (
                    (rows - x_width_center_px) ** 2.))
        dist = dist_px * dem_resolution  # I guess it is what they call s in Geiger

        # I should here save each profile that could later on be used (either saved in a dictionary or
        # directly save to a text file. I would prefer first to be saved in a dictionary and then
        # save to a text file) HERE MODIFY
        crossSections[i] = z_extracted[:]
        XSections[i] = cols
        YSections[i] = rows

        # ncol_1r and nrow_1r needs to be integer (This is dangerous as you
        # can have altitude similar to the rim height outside of the rim!
        # this needs to be fixed
        # much better to choose the nearest heights and widths in terms of
        # coordinates

        value_nearest, idx_nearest = find_nearest(z_extracted, z[ncol_1r,
                                                                 nrow_1r])

        diameter[i] = dist[idx_nearest] * 2.0

        # distance normalized
        dist_norm = dist / dist[idx_nearest]

        # geomorphometric calculations
        (depth[i], h[i], hr[i], cse[i], slope_mcw[i], slope_ucw[i],
         slope_fsa[i], slope_urs[i], slope_lrs[i], frdl[i], crdl[i],
         ucw_roc_val[i], uf_roc_val[i]) = geomorphometry_cs(z_extracted, dist, dist_norm)

    return (
    ucw_roc_val, uf_roc_val, cse, slope_mcw, slope_ucw, slope_fsa, slope_lrs,
    slope_urs, crdl, frdl,
    h, hr, depth, diameter, len(idx_circle2), cs_unique, crossSections,
    YSections, XSections)