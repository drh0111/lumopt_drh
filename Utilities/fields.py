import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from Utilities.scipy_wrappers import wrapped_GridInterpolator

class Fields():
    """
    This class act as a container for the field's data from field monitor. Several different interpolation objects are created to evaluate the field at any point in space. Use auxiliary method LUMOPT.Lumerical_methods.lumerical_scripts.get_fields to create this object
    """

    def __init__(self, x, y, z, wl, E, D = None, H = None, eps = None) -> None:
        """
        ---INPUTS---
        X:  1-D array. Represent the grids on x axis
        Y:  1-D array. Represent the grids on y axis
        Z:  1-D array. Represent the grids on z axis
        WL: 1-D array. Represent the grids on wavelength
        E:  Ndarray (x, y, z, wl, F). Represent the E field.
        D:  Ndarray (x, y, z, wl, F). Represent the D field.
        H:  Ndarray (x, y, z, wl, F). Represent the H field.
        EPS:    Ndarray (x, y, z, wl, F). Represent the dielectric constant.
        """
        def process_input(input):
            """precondition the INPUT so that it is proper array"""
            if type(input) is float:
                output = np.array([input])
            elif type(input) is np.ndarray:
                output = input.squeeze()
            else:
                raise UserWarning('input should be array')

            if output.shape == ():
                output = np.array([output])
            return output

        x, y, z, wl = map(process_input, [x, y, z, wl])

        self.x = x
        self.y = y
        self.z = z
        self.wl = wl
        self.E = E
        self.D = D
        self.H = H
        self.eps = eps
        self.pointing_vect = None
        self.normalized = False

        self.getfield = self.make_field_interpolation_object(E)
        if not (D is None):
            self.getDfield = self.make_field_interpolation_object(D)
        if not (H is None):
            self.getHfield = self.make_field_interpolation_object(H)
        if not (eps is None):
            self.geteps = self.make_field_interpolation_object(eps)

    def scale(self, dimension, factor):
        """
        This function is used to scale the E, D, and H field along the desired dimension with the given factor
        ---INPUTS---
        DIMENSION:  Integer
            The dimension to be scaled, 0 (x-axis), 1 (y-axis), 2 (z-axis), 3 (wavelength), 4 (vector field).
        FACTOR:     Array
            The dimension equals the dimension to be scaled
        """
        if hasattr(self.E, 'dtype'): # make sure it exist and is array
            if self.E.shape[dimension] == len(factor):
                self.E = np.concatenate([np.take(self.E, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getfield = self.make_field_interpolation_object(self.E)
            else:
                raise UserWarning('the length of factor does not fit the E field')
        if hasattr(self.D, 'dtype'):
            if self.D.shape[dimension] == len(factor):
                self.D = np.concatenate([np.take(self.D, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getDfield = self.make_field_interpolation_object(self.D)
            else:
                raise UserWarning('the length of factor does not fit the D field')
        if hasattr(self.H, 'dtype'):
            if self.H.shape[dimension] == len(factor):
                self.H = np.concatenate([np.take(self.H, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getfield = self.make_field_interpolation_object(self.H)
            else:
                raise UserWarning('the length of factor does not fit the H field')

    def make_field_interpolation_object(self, F):
        """
        This function is used to make a linear-interpolated field from the original one from the field monitor
        ---INPUT---
        F: Ndarray, the original field to be interpolated.
        """
        # Problem: what if not x, y, z?
        wl = self.wl[0] if len(self.wl) > 1 and F.shape[3] == 1 else self.wl

        Fx_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z, wl), F[:, :, :, :, 0], bounds_error = False)
        Fy_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z, wl), F[:, :, :, :, 1], bounds_error = False)
        Fz_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z, wl), F[:, :, :, :, 2], bounds_error = False)

        def field_interpolator(x, y, z, wl):
            """interpolated function"""
            Fx = Fx_interpolator((x, y, z, wl))
            Fy = Fy_interpolator((x, y, z, wl))
            Fz = Fz_interpolator((x, y, z, wl))

            return np.array([Fx, Fy, Fz]).squeeze()
        
        return field_interpolator

class FieldsNoInterp(Fields):
    """
    This class of container will should be used when the interpolation option of Lumerical FDTD is diabled, which means Lumerical will not interpolate the field (because Yee grid is staggered grid) to get the field data at the same position.
    """
    def __init__(self, x, y, z, wl, deltas, E, D=None, H=None, eps=None) -> None:
        def process_input(input):
            """precondition the INPUT so that it is proper array"""
            if type(input) is float:
                output = np.array([input])
            elif type(input) is np.ndarray:
                output = input.flatten()
            else:
                raise UserWarning('input should be array')

            if output.shape == ():
                output = np.array([output])
            return output

        delta_x = deltas[0]
        delta_y = deltas[1]
        delta_z = deltas[2]

        x, y, z, wl, delta_x, delta_y, delta_z = map(process_input, [x, y, z, wl, delta_x, delta_y, delta_z])
        deltas = [delta_x, delta_y, delta_z]

        self.x = x
        self.y = y
        self.z = z
        self.wl = wl
        self.deltas = deltas
        self.E = E
        self.D = D
        self.H = H
        self.eps = eps
        self.pointing_vect = None
        self.normalized = False

        self.getfield = self.make_field_interpolation_object_nointerp(E)
        if not D is None:
            self.getDfield = self.make_field_interpolation_object_nointerp(D)
        if not eps is None:
            self.geteps = self.make_field_interpolation_object_nointerp(eps)
        if not H is None:
            self.getHfield = self.make_field_interpolation_object(H) # Problem: H position
        self.evals = 0

    def make_field_interpolation_object_nointerp(self, F):
        """
        This function make some position shift to fit the Lumerical's nointerpolation option, and make the linear-interpolation to the raw nointerpolated data from the field monitor. Be aware that it only fit the E, D and eps field
        ---INPUT---
        F: Ndarray, the original field to be interpolated.
        """

        wl =  self.wl[0] if len(self.wl) > 1 and F.shape[3] == 1 else self.wl

        Fx_interpolator = wrapped_GridInterpolator((self.x + self.deltas[0], self.y, self.z, wl), F[:, :, :, :, 0], bounds_error = False)
        Fy_interpolator = wrapped_GridInterpolator((self.x, self.y + self.deltas[1], self.z, wl), F[:, :, :, :, 1], bounds_error = False)
        Fz_interpolator = wrapped_GridInterpolator((self.x, self.y, self.z + self.deltas[2], wl), F[:, :, :, :, 2], bounds_error = False)

        def field_interpolator(x, y, z, wl):
            """interpolated function"""
            Fx = Fx_interpolator((x, y, z, wl))
            Fy = Fy_interpolator((x, y, z, wl))
            Fz = Fz_interpolator((x, y, z, wl))

            return np.array([Fx, Fy, Fz]).squeeze()

        return field_interpolator

    def scale(self, dimension, factor):
        """
        This function is used to scale the E, D, and H field along the desired dimension with the given factor
        ---INPUTS---
        DIMENSION:  Integer
            The dimension to be scaled, 0 (x-axis), 1 (y-axis), 2 (z-axis), 3 (wavelength), 4 (vector field).
        FACTOR:     Array
            The dimension equals the dimension to be scaled
        """
        if hasattr(self.E, 'dtype'): # make sure it exist and is array
            if self.E.shape[dimension] == len(factor):
                self.E = np.concatenate([np.take(self.E, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getfield = self.make_field_interpolation_object_nointerp(self.E)
            else:
                raise UserWarning('the length of factor does not fit the E field')
        if hasattr(self.D, 'dtype'):
            if self.D.shape[dimension] == len(factor):
                self.D = np.concatenate([np.take(self.D, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getDfield = self.make_field_interpolation_object_nointerp(self.D)
            else:
                raise UserWarning('the length of factor does not fit the D field')
        if hasattr(self.H, 'dtype'):
            if self.H.shape[dimension] == len(factor):
                self.H = np.concatenate([np.take(self.H, [index], axis = dimension) for index in range(len(factor))] , axis = dimension)
                self.getfield = self.make_field_interpolation_object(self.H)
            else:
                raise UserWarning('the length of factor does not fit the H field')