import pandas as pd
import geopandas as gpd
import numpy as np
import scipy
import scipy.ndimage
import sys

sys.path.append("/home/nilscp/GIT/crater_morphometry")
from pathlib import Path
from scipy.optimize import curve_fit
from rim_detection import rim_detection
from tqdm import tqdm

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
               ucw_roc, uf_roc, frdl, crdl, slope_urs, slope_lrs)
    else:
        return(radius, min_depth, center_depth, mcw, ucw, fsa, cse, rim_height_absolute, rim_height_relative)

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

def main_ellipse(location_of_craters, shp_folder, dem_folder, out_folder, suffix, prefix, craterID=None):

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

    radi_min, radi_25p, radi_med, radi_75p, radi_max, radi_std, radi_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    mdep_min, mdep_25p, mdep_med, mdep_75p, mdep_max, mdep_std, mdep_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    mcwa_min, mcwa_25p, mcwa_med, mcwa_75p, mcwa_max, mcwa_std, mcwa_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    ucwa_min, ucwa_25p, ucwa_med, ucwa_75p, ucwa_max, ucwa_std, ucwa_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    fsan_min, fsan_25p, fsan_med, fsan_75p, fsan_max, fsan_std, fsan_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    csex_min, csex_25p, csex_med, csex_75p, csex_max, csex_std, csex_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    rhab_min, rhab_25p, rhab_med, rhab_75p, rhab_max, rhab_std, rhab_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    rhre_min, rhre_25p, rhre_med, rhre_75p, rhre_max, rhre_std, rhre_ste = [np.zeros(gdf_selection.shape[0]) for _ in range(7)]
    index = np.zeros(gdf_selection.shape[0])

    i = 0
    for ind, row in tqdm(gdf_selection.iterrows(), total=gdf_selection.shape[0]):
        ellipse_shp = shp_dummy.with_name(row.CRATER_ID + suffix.split(".")[0] + "_ellipse_candidate2_polygon.shp")
        crater_dem_detrended = dem_dummy.with_name(row.CRATER_ID + suffix)
        data = geomorphometry_cross_section_ellipse(ellipse_shp, crater_dem_detrended, is_extra_calc=False)
        radi_min[i], radi_25p[i], radi_med[i], radi_75p[i], radi_max[i], radi_std[i], radi_ste[i] = [r for r in statistic(data[0], data[0].shape[0])]
        mdep_min[i], mdep_25p[i], mdep_med[i], mdep_75p[i], mdep_max[i], mdep_std[i], mdep_ste[i] = [r for r in statistic(data[1], data[0].shape[0])]
        mcwa_min[i], mcwa_25p[i], mcwa_med[i], mcwa_75p[i], mcwa_max[i], mcwa_std[i], mcwa_ste[i] = [r for r in statistic(data[3], data[0].shape[0])]
        ucwa_min[i], ucwa_25p[i], ucwa_med[i], ucwa_75p[i], ucwa_max[i], ucwa_std[i], ucwa_ste[i] = [r for r in statistic(data[4], data[0].shape[0])]
        fsan_min[i], fsan_25p[i], fsan_med[i], fsan_75p[i], fsan_max[i], fsan_std[i], fsan_ste[i] = [r for r in statistic(data[5], data[0].shape[0])]
        csex_min[i], csex_25p[i], csex_med[i], csex_75p[i], csex_max[i], csex_std[i], csex_ste[i] = [r for r in statistic(data[6], data[0].shape[0])]
        rhab_min[i], rhab_25p[i], rhab_med[i], rhab_75p[i], rhab_max[i], rhab_std[i], rhab_ste[i] = [r for r in statistic(data[7], data[0].shape[0])]
        rhre_min[i], rhre_25p[i], rhre_med[i], rhre_75p[i], rhre_max[i], rhre_std[i], rhre_ste[i] = [r for r in statistic(data[8], data[0].shape[0])]
        index[i] = row['index']
        i = i + 1

    # create a pandas dataframe
    lst = np.column_stack([radi_min, radi_25p, radi_med, radi_75p, radi_max, radi_std, radi_ste,
           mdep_min, mdep_25p, mdep_med, mdep_75p, mdep_max, mdep_std, mdep_ste,
           mcwa_min, mcwa_25p, mcwa_med, mcwa_75p, mcwa_max, mcwa_std, mcwa_ste,
           ucwa_min, ucwa_25p, ucwa_med, ucwa_75p, ucwa_max, ucwa_std, ucwa_ste,
           fsan_min, fsan_25p, fsan_med, fsan_75p, fsan_max, fsan_std, fsan_ste,
           csex_min, csex_25p, csex_med, csex_75p, csex_max, csex_std, csex_ste,
           rhab_min, rhab_25p, rhab_med, rhab_75p, rhab_max, rhab_std, rhab_ste,
           rhre_min, rhre_25p, rhre_med, rhre_75p, rhre_max, rhre_std, rhre_ste])

    df = pd.DataFrame(lst, index=index.astype('int32'), columns=geormorph_columns())
    crater_id = [prefix + str(index).zfill(4) for index, row in df.iterrows()]
    df["CRATER_ID"] = crater_id

    # calculate diam
    diam_min, diam_25p, diam_med, diam_75p, diam_max, diam_std, diam_ste = [array * 2.0 for array in [radi_min, radi_25p, radi_med, radi_75p, radi_max, radi_std, radi_ste]]
    df["diam_min"] = diam_min
    df["diam_25p"] = diam_25p
    df["diam_med"] = diam_med
    df["diam_75p"] = diam_75p
    df["diam_max"] = diam_max
    df["diam_std"] = diam_std
    df["diam_ste"] = diam_ste

    # calculate depth-diam
    dedi_min, dedi_25p, dedi_med, dedi_75p, dedi_max = [array for array in [(rhab_min-mdep_min)/diam_min, (rhab_25p-mdep_25p)/diam_25p, (rhab_med-mdep_med)/diam_med, (rhab_75p-mdep_75p)/diam_75p, (rhab_max-mdep_max)/diam_max]]
    df["dedi_min"] = dedi_min
    df["dedi_25p"] = dedi_25p
    df["dedi_med"] = dedi_med
    df["dedi_75p"] = dedi_75p
    df["dedi_max"] = dedi_max


    # Merge it to the geopandas dataframe.

    # Eventually add information such as within_mare and freshness

def statistic(array, n):
    percentile = [0.0, 25.0, 50.0, 75.0, 100.0]
    arr_min, arr_25p, arr_med, arr_75p, arr_max = [np.percentile(array, p) for p in percentile]
    std = np.std(array)
    std_error = std / np.sqrt(n)
    return (arr_min, arr_25p, arr_med, arr_75p, arr_max, std, std_error)


def geormorph_columns():

    columns = ["diam_min", "diam_25p", "diam_med", "diam_75p", "diam_max", "diam_std", "diam_ste",
               "mdep_min", "mdep_25p", "mdep_med", "mdep_75p", "mdep_max", "mdep_std", "mdep_ste",
               "mcwa_min", "mcwa_25p", "mcwa_med", "mcwa_75p", "mcwa_max", "mcwa_std", "mcwa_ste",
               "ucwa_min", "ucwa_25p", "ucwa_med", "ucwa_75p", "ucwa_max", "ucwa_std", "ucwa_ste",
               "fsan_min", "fsan_25p", "fsan_med", "fsan_75p", "fsan_max", "fsan_std", "fsan_ste",
               "csex_min", "csex_25p", "csex_med", "csex_75p", "csex_max", "csex_std", "csex_ste",
               "rhab_min", "rhab_25p", "rhab_med", "rhab_75p", "rhab_max", "rhab_std", "rhab_ste",
               "rhre_min", "rhre_25p", "rhre_med", "rhre_75p", "rhre_max", "rhre_std", "rhre_ste"]

    return (columns)


def is_on_mare(location_of_craters, mare_shp):

    gdf = gpd.read_file(location_of_craters)
    gdf_mare = gpd.read_file(mare_shp)
    gdf_mare_proj = gdf_mare.to_crs(gdf.crs.to_wkt())
    gdf_expl = gdf_mare_proj.explode()
    mare_geom = gdf_expl.geometry.unary_union


    is_mare = []
    for index, row in gdf.iterrows():
        if row.geometry.within(mare_geom):
            is_mare.append(1)
        else:
            is_mare.append(0)