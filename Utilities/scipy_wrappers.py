from scipy.interpolate import RegularGridInterpolator
import numpy as np

def wrapped_GridInterpolator(points, values, method = 'linear', bounds_error = True, fill_value = float('nan')):
    """
    This function is used to expand the functionality of RegularGridInterpolator so that it can deal with 1 size dimension (dimension which only have 1 number)

    ---INPUTS---
    POINTS: Tuple of ndarray of float, with shapes (m1, ), ..., (mn, ).  
        The points defining the regular grid in n dimensions.
    VALUES: Ndarray, with shape (m1, ..., mn). 
        The values on each grid.
    METHOD: String, optional
        The method of interpolation to perform. 'linear' and 'nearest' are supported. This parameter will be default for the object's '__call__' method. Default is 'linear'
    BOUNDS_ERROR:   Bool, optional
        If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used.
    FILL_VALUE: number, optional
        If provided, the value to use for points outside of the interpolation domain. If None, values outside the domain are extrapolated.
    ---OUTPUT---
    WRAPPED_INTERPOLATOR:   Callable function.
        It take arrays of points as input and return array of interpolated values
    """

    # Find the dimensions with 1 number, and modify points and values
    newpoints = []
    dim_1_input = [point.size == 1 for point in points]
    non_dim_1_input = [point.size != 1 for point in points]
    for point, flag in zip(points, dim_1_input):
        if flag == False:
            newpoints.append(point)
    newvalues = values.copy().squeeze()
    # Flag of only one grid
    singleton = newpoints == [] 

    if singleton == True:
        def wrapped_interpolator(points):
            return newvalues
    else:
        interpolator = RegularGridInterpolator(tuple(newpoints), newvalues, method, bounds_error, fill_value)

        def wrapped_interpolator(points):
            """
            deal with those redundant dimensions and return the interpolated values of the given points
            ---INPUT---
            POINTS: 
                1) 2D array with shape (n, m) or array of tuples. m is the dimension of each point, this stands for the try case
                2) tuple of array or list representing the coordinates of x, y, z... for the desired points, this stands for the except case
            """
            # Considering the difference ponits = [(x0, y0, z0...), ...] and points = [[x0, x1..., xm], [y0, y1..., ym], ...]
            try:
                newpoints = []

                for point in points:
                    newpoint = []
                    for coord, dim_1_flag in zip(point, dim_1_input):
                        if dim_1_flag == False:
                            if type(coord) is np.array or type(coord) is list:
                                newpoint.append(coord[0])
                            else:
                                newpoint.append(coord)
                    newpoints.append(tuple(newpoint))
                return interpolator(np.array(newpoints))
            except:
                newpoint = []
                for coord, dim_1_flag in zip(points, dim_1_input):
                    if dim_1_flag == False:
                        if type(coord) is np.array or type(coord) is list:
                            newpoint.append(coord[0])   # Problem: different from standard one
                        else:
                            newpoint.append(coord)
                return interpolator(tuple(newpoint))

        return wrapped_interpolator