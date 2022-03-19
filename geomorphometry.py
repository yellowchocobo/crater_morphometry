import numpy as np
import scipy
import scipy.ndimage
import sys

sys.path.append("/home/nilscp/GIT/crater_morphometry")
from scipy.optimize import curve_fit
from rim_detection import rim_detection

def argmin_values_along_axis(arr, value, axis):
    argmin_idx = np.abs(arr - value).argmin(axis=axis)
    shape = arr.shape
    indx = list(np.ix_(*[np.arange(i) for i in shape]))
    indx[axis] = np.expand_dims(argmin_idx, axis=axis)
    return (np.squeeze(arr[tuple(indx)]), argmin_idx)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)

def calc_R(x,y, xc, yc):
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def calc_D(c, x, y):
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    xm = np.mean(x)
    ym = np.mean(y)
    center_estimate = xm, ym
    
    center, ier = scipy.optimize.leastsq(calc_D, center_estimate, args=(x,y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def power(x,a,b,c):   
    return a * (x**b) + c

def linear(x,a,b):
    return a*x + b

def normalized_radius(distances,radius):
    return np.divide(distances,radius,out=np.zeros_like(distances), where=radius!=0)

def geomorphometry_cross_section(ellipse_shp, crater_dem_detrended, is_extra_calc=False):

    (z_detrended, yc_px, xc_px,
     y_px_ell, x_px_ell, y_px_ell2, x_px_ell2,
     crater_radius_px, dem_resolution) = rim_detection.load_ellipse(ellipse_shp, crater_dem_detrended)

    y_heights_px_ell, x_widths_px_ell, z_profiles_ell, distances_ell, cross_sections_ids_ell = rim_detection.extract_cross_sections_from_ellipse(
        crater_radius_px, dem_resolution, y_px_ell2, x_px_ell2, yc_px,
        xc_px, z_detrended)

    # the middle of the cross-sections is the rim of the crater
    n = np.int32(np.round(x_widths_px_ell.shape[1] / 2.0))

    x_1R = x_widths_px_ell[:, n]
    y_1R = y_heights_px_ell[:, n]
    h_d = (y_1R - yc_px) ** 2.0
    w_d = (x_1R - xc_px) ** 2.0
    radius = np.sqrt(h_d + w_d) * dem_resolution

    # small variations but almost the same across all cross-sections
    radius_norm = np.apply_along_axis(normalized_radius, 0, distances_ell, radius)

    # It should actually be the same index for all cross sections
    A, idxA = argmin_values_along_axis(arr=radius_norm, value=0.0, axis=1)
    B, idxB = argmin_values_along_axis(arr=radius_norm, value=0.1, axis=1)
    C, idxC = argmin_values_along_axis(arr=radius_norm, value=0.7, axis=1)
    D, idxD = argmin_values_along_axis(arr=radius_norm, value=0.8, axis=1)
    E, idxE = argmin_values_along_axis(arr=radius_norm, value=0.9, axis=1)
    F, idxF = argmin_values_along_axis(arr=radius_norm, value=1.0, axis=1)
    G, idxG = argmin_values_along_axis(arr=radius_norm, value=1.2, axis=1)
    H, idxH = argmin_values_along_axis(arr=radius_norm, value=2.0, axis=1)

    # middle cavity slope - C (0.7R) to E (0.9R)
    mcw = cavity_slope(distances_ell, z_profiles_ell, idxC[0], idxE[0])

    # upper cavity slope - D (0.8R) to F (1.0R)
    ucw = cavity_slope(distances_ell, z_profiles_ell, idxD[0], idxF[0])

    # Flank slope angle - F (1.0R) to G (1.2R)
    fsa = np.abs(cavity_slope(distances_ell, z_profiles_ell, idxF[0], idxG[0]))

    # absolute rim height
    rim_height_absolute = np.array([z_prof[idxF[0]] for z_prof in z_profiles_ell])

    # relative rim height
    rim_height_relative = np.abs(np.array([z_prof[idxF[0]] - np.min(z_prof[idxF[0]:idxH[0]]) for z_prof in z_profiles_ell]))

    # mininimum depth
    min_depth = np.array([np.min(z_prof[idxA[0]:idxF[0]+1]) for z_prof in z_profiles_ell])

    # center depth
    center_depth = z_profiles_ell[0][0]

    # cavity shape exponent   - F (1.0R) to G (1.2R) -  can contain NaN!
    cse = np.array([cavity_shape_exponent(distances_ell[i][idxB[0]:idxE[0] + 1],
                                              z_profiles_ell[i][idxB[0]:idxE[0] + 1]) for i in np.arange(z_profiles_ell.shape[0])])

    if is_extra_calc:
        # upper cavity wall roc  - D (0.8R) to F (1.0R) - radius_of_curvature
        ucw_roc =  np.array([radius_of_curvature(distances_ell[i][idxD[0]:idxF[0]+1],
                                          z_profiles_ell[i][idxD[0]:idxF[0]+1]) for i in
                         np.arange(z_profiles_ell.shape[0])])

        # upper flank roc  - F (1.0R) to G (1.2R) -  radius_of_curvature
        uf_roc =  np.array([radius_of_curvature(distances_ell[i][idxF[0]:idxG[0]+1],
                                          z_profiles_ell[i][idxF[0]:idxG[0]+1]) for i in
                         np.arange(z_profiles_ell.shape[0])])

        # flank rim decay  - F (1.0R) to H (2.0R) -  length rim_decay_length
        frdl = np.array([rim_decay_length(distances_ell[i][idxF[0]:idxH[0]],
                                          z_profiles_ell[i][idxF[0]:idxH[0]],
                                          radius[i]) for i in
                         np.arange(z_profiles_ell.shape[0])])

        # cavity rim decay  - A (0.0R) to F (1.2R) -  length rim_decay_length
        crdl = np.array([rim_decay_length(distances_ell[i][idxA[0]:idxF[0] + 1],
                                          z_profiles_ell[i][
                                          idxA[0]:idxF[0] + 1],
                                          radius[i]) for i in
                         np.arange(z_profiles_ell.shape[0])])

        # upper rim span
        slope_urs = 180. - (ucw + fsa)

        # lower rim span
        slope_lrs = 180. - (mcw + fsa)

        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative,
               ucw_roc, uf_roc, frdl, crdl, slope_urs, slope_lrs)
    else:
        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative)


def radius_of_curvature(distances, z_profile):
    try:
        __, __, roc_val, __ = leastsq_circle(distances, z_profile)
            
    except:
        roc_val = np.nan
        
    return (roc_val)

def rim_decay_length(distances, z_profile, radius):
    slope_deg = np.abs(np.rad2deg(np.arctan(np.gradient(z_profile,distances))))
    slope_max = np.max(slope_deg)
    __, idx_frdl = find_nearest(slope_deg, slope_max / 2.0)
    rdl = np.abs(distances[idx_frdl] - radius)
    return rdl

def cavity_shape_exponent(distances, z_profile):
    try:
        a, b = curve_fit(power, distances, z_profile, po=[1e-3,1.5,-1000], maxfev=10000)
        exponent = a[1]
        cse = exponent
    except:
        cse = np.nan
    return (cse)

def cavity_slope(distances, z_profiles, starting_index, end_index):

    slope_deg = np.array([
        np.rad2deg(
            np.arctan(
                np.polyfit(distances[i][starting_index:end_index+1],
                       z_profiles[i][starting_index:end_index+1],1)[0])) for i in range(z_profiles.shape[0])])

    return slope_deg


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

    # This step can be avoided can use the predict of the ellipsoid with
    # 2 a and 2b (will this be correct?)

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