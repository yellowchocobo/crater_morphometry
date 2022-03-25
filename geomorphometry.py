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

def geomorphometry_cross_section_pts(point_shp, ellipse_shp, crater_dem_detrended, inner_t, outer_t, is_extra_calc=False):

    (z_detrended, yc_px, xc_px,
     y_px_pts, x_px_pts, y_px_pts_2R, x_px_pts_2R,
     dem_resolution, crater_radius_px) = rim_detection.load_pts(point_shp, ellipse_shp, crater_dem_detrended, inner_t, outer_t)

    y_heights_px_pts, x_widths_px_pts, z_profiles_pts, distances_pts, radius, cross_sections_ids_pts = rim_detection.extract_cross_sections_from_pts(
        crater_radius_px, dem_resolution, y_px_pts, x_px_pts, y_px_pts_2R, x_px_pts_2R, yc_px,
        xc_px, z_detrended)

    # small variations but almost the same across all cross-sections
    radius_norm = np.apply_along_axis(normalized_radius, 0, distances_pts, radius)

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
    mcw = cavity_slope(distances_pts, z_profiles_pts, idxC[0], idxE[0])

    # upper cavity slope - D (0.8R) to F (1.0R)
    ucw = cavity_slope(distances_pts, z_profiles_pts, idxD[0], idxF[0])

    # Flank slope angle - F (1.0R) to G (1.2R)
    fsa = np.abs(cavity_slope(distances_pts, z_profiles_pts, idxF[0], idxG[0]))

    # absolute rim height
    rim_height_absolute = np.array([z_prof[idxF[0]] for z_prof in z_profiles_pts])

    # relative rim height
    rim_height_relative = np.abs(np.array([z_prof[idxF[0]] - np.min(z_prof[idxF[0]:idxH[0]]) for z_prof in z_profiles_pts]))

    # mininimum depth
    min_depth = np.array([np.min(z_prof[idxA[0]:idxF[0]+1]) for z_prof in z_profiles_pts])

    # center depth
    center_depth = z_profiles_pts[0][0]

    # cavity shape exponent   - F (1.0R) to G (1.2R) -  can contain NaN!
    cse = np.array([cavity_shape_exponent(distances_pts[i][idxB[0]:idxE[0] + 1],
                                              z_profiles_pts[i][idxB[0]:idxE[0] + 1]) for i in np.arange(z_profiles_pts.shape[0])])
    # Number of observations
    nc = y_heights_px_pts.shape[0]

    if is_extra_calc:
        # upper cavity wall roc  - D (0.8R) to F (1.0R) - radius_of_curvature
        ucw_roc =  np.array([radius_of_curvature(distances_pts[i][idxD[0]:idxF[0]+1],
                                          z_profiles_pts[i][idxD[0]:idxF[0]+1]) for i in
                         np.arange(z_profiles_pts.shape[0])])

        # upper flank roc  - F (1.0R) to G (1.2R) -  radius_of_curvature
        uf_roc =  np.array([radius_of_curvature(distances_pts[i][idxF[0]:idxG[0]+1],
                                          z_profiles_pts[i][idxF[0]:idxG[0]+1]) for i in
                         np.arange(z_profiles_pts.shape[0])])

        # flank rim decay  - F (1.0R) to H (2.0R) -  length rim_decay_length
        frdl = np.array([rim_decay_length(distances_pts[i][idxF[0]:idxH[0]],
                                          z_profiles_pts[i][idxF[0]:idxH[0]],
                                          radius[i]) for i in
                         np.arange(z_profiles_pts.shape[0])])

        # cavity rim decay  - A (0.0R) to F (1.2R) -  length rim_decay_length
        crdl = np.array([rim_decay_length(distances_pts[i][idxA[0]:idxF[0] + 1],
                                          z_profiles_pts[i][
                                          idxA[0]:idxF[0] + 1],
                                          radius[i]) for i in
                         np.arange(z_profiles_pts.shape[0])])

        # upper rim span
        slope_urs = 180. - (ucw + fsa)

        # lower rim span
        slope_lrs = 180. - (mcw + fsa)

        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative,
               ucw_roc, uf_roc, frdl, crdl, slope_urs, slope_lrs, nc)
    else:
        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative, nc)

def geomorphometry_cross_section_ellipse(ellipse_shp, crater_dem_detrended, is_extra_calc=False):

    (z_detrended, yc_px, xc_px,
     y_px_ell, x_px_ell, y_px_ell2, x_px_ell2,
     dem_resolution, crater_radius_px) = rim_detection.load_ellipse(ellipse_shp, crater_dem_detrended)

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
    # Number of observations
    nc = 512

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
               ucw_roc, uf_roc, frdl, crdl, slope_urs, slope_lrs, nc)
    else:
        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative, nc)


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
        a, b = curve_fit(power, distances, z_profile, p0=[1e-3,1.5,-1000], maxfev=10000)
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