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