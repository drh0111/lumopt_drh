import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import lumapi
from Geometries.geometry import Geometry
from Lumerical_methods.lumerical_scripts import set_spatial_interp

eps0 = sp.constants.epsilon_0 # Problem: is it correct?

class Density2D(Geometry):
    """
    This class is a parametrization of geometry commonly used in topology optimization named density method, and we only consider 2D cases here, where variables Z, DEPTH are fixed (not direct variables), and the FDTD simulation, monitors are 2D.
    """

    def __init__(self, eps, x, y, z, eps_min, eps_max):
        """
        ---INPUTS---
        EPS: 2-D numpy array
            An M*N array, representing the epsilon of the material in given positions of the design region
        X:  1-D numpy array
            An M element array, it represents the x positions of the grids in 'm' unit, and must be equally spaced
        Y:  1-D numpy array
            An N element array, it represents the y positions of the grids in 'm' unit, and must be equally spaced
        Z:  1-D numpy array
            An P element array, it represents the Z positions of the grids in 'm' unit, and must be equally spaced
        EPS_MIN:    Real value
            The minimum epsilon value in the given structure
        EPS_MAX:    Real value
            The maximum epsilon value in the given structure
        ---PARAMS---
        DENSITY:    2-D numpy array
            An M*N array, and it's element value vary from 0 to 1, representing the density of the material in given position        
        PARAMS: 1-D array
            aN 1-d array with M*N elements, it is the flattened array of DENSITY in row-major (C-style) convention, to fit with optimizer.
        BOUNDS: 2-D array
            An (M*N)*2 array, it represents the bounds of every parameters, which is flattened into 1D array (as for M*N) in row-major (C-style) convention.
        SIZE: Tuple of integers
            Tuple representing the size of EPS array
        DX: Real value
            Grid length in x direction in 'm' unit
        DY: Real value
            Grid length in y direction in 'm' unit    
        DZ: Real value
            Grid length in z direction in 'm' unit
        DEPTH:  Real value
            The depth of the structure (z direction) in 'm' unit
        EPS_HIST:   List
            containing the history of epsilon in the optimization
        COLORBAR:   None or Colorbar object
            Handle of epsilon distribution's colorbar, used to update colorbar
        """

        self.eps = eps
        self.x = x
        self.y = y
        self.z = (z[0]+z[-1])/2
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.bounds = np.array([(0, 1)] * np.size(eps)) # () is used to adjust the shape
        self.size = eps.shape
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]
        self.depth = z[-1] - z[0]
        self.eps_hist = []
        self.density = self.from_eps_to_density(eps) 
        self.params = self.density.flatten()
        self.colorbar = None

        self.unfold_symmetry = True # Do not let monitor unfold symmetry

    def use_interpolation(self):
        return True

    def from_eps_to_density(self, eps):
        """
        This function is used to convert variable EPS to variable DENSITY following the desired interpolation scheme
        """
        # Problem: should be staticmethod here?
        return (eps-self.eps_min) / (self.eps_max-self.eps_min)

    def from_density_to_eps(self, density):
        """
        This function is used to convert variable DENSITY to variable EPS following the desired interpolation scheme
        """
        # Problem: should be staticmethod here?
        return self.eps_min + (self.eps_max-self.eps_min)*density
    
    # def reform_shape(self, variable):
    #     """
    #     This function is used to convert 'PARAMS' shaped variable to 'DENSITY' shaped one, and vice versus
    #     ---INPUT---
    #     VARIABLE:   1D or 2D array
    #         1D PARAMS shaped variable or 2D DENSITY shaped variable, with number consistent with EPSILON
    #     """
    #     assert variable.size == self.eps.size, 'Size of variable incompatible with epsilon'
    #     if variable.ndim == 1:
    #         return np.reshape(variable, self.size)
    #     elif variable.ndim == 2:
    #         return np.reshape(variable, (-1))
    #     else:
    #         raise UserWarning("Input array's dimensionality is invalid")
        
    def update_geometry(self, params):
        """
        This function is used to update and record the density and epsilon of the geometry during the optimization in python (not Lumerical simulation yet)
        ---INPUT---
         PARAMS:  Must be 1 dimensional array or None. 
            It is the epsilon of the added/updated geometry aligned in 1 dimension in row-major (C-style) convention. (It is 1D to fit with optimizer)
        """
        self.params = params
        self.density = np.reshape(params, self.size)
        self.eps = self.from_density_to_eps(self.density)
        self.eps_hist.append(self.eps)


    def add_geo(self, sim, params, only_update):
        """
        This function will add and update the geometry in the Lumerical simulation

        ---INPUTS---
         PARAMS:  Must be 1 dimensional array or None. 
            It is the epsilon of the added/updated geometry aligned in 1 dimension in row-major (C-style) convention. (It is 1D to fit with optimizer)
        """

        sim.fdtd.switchtolayout()
        eps = self.eps if params is None else self.from_density_to_eps(np.reshape(params, self.size))

        sim.fdtd.putv('x_geo', self.x)
        sim.fdtd.putv('y_geo', self.y)
        # The import geometry will be 2D here
        sim.fdtd.putv('z_geo', np.array([self.z - self.depth/2, self.z + self.depth/2]))
        
        if not only_update:
            # set the monitors and fdtd mesh to fit the desired geometry settings
            set_spatial_interp(sim.fdtd, 'opt_fields', 'specified position')
            set_spatial_interp(sim.fdtd, 'opt_fields_index', 'specified position')

            script = ('select("opt_fields");'
                      'set("x min", {});'
                      'set("x max", {});'
                      'set("y min", {});'
                      'set("y max", {});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y))
            sim.fdtd.eval(script)

            script = ('select("opt_fields_index");'
                      'set("x min", {});'
                      'set("x max", {});'
                      'set("y min", {});'
                      'set("y max", {});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y))
            sim.fdtd.eval(script)

            sim.fdtd.eval('addimport;' + 'set("detail", 1);')

            mesh_script = ('addmesh;'
                           'set("x min", {});'
                           'set("x max", {});'
                           'set("y min", {});'
                           'set("y max", {});'
                           'set("dx", {});'
                           'set("dy", {});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y), self.dx, self.dy)
            sim.fdtd.eval(mesh_script)

        if eps is not None:
            sim.fdtd.putv('eps_geo', eps)

            # Delete and re-add the import to avoid a warning
            script = ('select("import");'
                      'delete;'
                      'addimport;'
                      'tmp = zeros(length(x_geo), length(y_geo), 2);'
                      'tmp(:, :, 1) = eps_geo;'
                      'tmp(:, :, 2) = eps_geo;'
                      'importnk2(sqrt(tmp), x_geo, y_geo, z_geo);')
            sim.fdtd.eval(script)

    def get_current_params(self):
        """
        This function is used to get the current value of parameters (in 1D array), it is consistent with of status of 'UPDATE_GEO' function
        """

        return self.params

    def calculate_gradients(self, gradient_fields):
        """
        This function is used to calculate gradients of FOM versus every density value (considering the different wavelength), given the gradient fields. Be aware that because we adjust mesh according to size of epsilon, we don't need to consider the rearrangement of GRADIENT_FIELDS and EPSILON to calculate gradients.

        ---INPUT---
        GRADIENT_FIELDS:    GradientField class
        ---PARAM---
        DX: Scalar
            The step size for direct differentiation
        ---OUTPUT---
        GRADIENTS:  2D array
            Arrary with shape (num_params, num_wl), representing the gradients vs param & wl
        """

        dx = 1e-9
        # Consider the wavelength dimesion (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        E_forward_dot_adjoint = np.atleast_3d(np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),-1))))

        dF_dEps = 2 * eps0 * self.dx * self.dy * E_forward_dot_adjoint

        # Use direct differentiation (dx = 1e-9) to calculate gradients. There might exist more convenient method for specific cases

        # The most general form
        # gradients = []
        # for i, param in self.params:
        #     d_param = self.params
        #     d_param[i] = self.params[i] + dx
        #     d_eps = np.atleast_3d(self.from_density_to_eps(np.reshape(d_param, self.size)) - self.eps)
        #     gradients.append(np.sum(d_eps * dF_dEps, (0,1)))

        # Consider the DENSITY - EPSILON process is element wise
        d_param = self.params + dx
        d_eps = np.atleast_3d(self.from_density_to_eps(np.reshape(d_param, self.size)) - self.eps)
        gradients = np.reshape(d_eps*dF_dEps, (d_eps.size, -1))

        return np.array(gradients)

    def plot(self, ax):
        """
        This function is used to plot the epsilon distribution in the given axe
        ---INPUT---
        AX: Axe object. The axe that the epsilon distribution will be plotted on
        """

        ax.clear()
        x = self.x
        y = self.y
        eps = self.eps
        image = ax.imshow(np.real(np.transpose(eps)), vmin = self.eps_min, vmax = self.eps_max, extent = [min(x)*1e6, max(x)*1e6, min(y)*1e6, max(y)*1e6], origin = 'lower')
        
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = plt.colorbar(mappable =  image, ax = ax)
        ax.set_title('Eps')
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        return True
