import sys
from tkinter import Y
import numpy as np
import scipy as sp
import random

import lumapi
from Geometries.geometry import Geometry
from Utilities.materials import Material
from Utilities.edge import Edge

class Polygon(Geometry):
    """
    This class of object is used to specify, generate and modify the polygon-typed geometry in Lumerical FDTD simulation. The definition of geometry is done by specifying the vertices on the x-y plane, which will be extruded along z axis to generate the full 3-D geometry.
    Be aware that the vertices should be given in a counter-clockwise direction, and the format of the vertices is np.array([(x0, y0), ..., (xn, yn)])

    ---INPUTS---
    POINTS: Array of shape (N, 2), positions of N polygon vertices
    Z: Scalar, the z position of the central x-y plane of polygon
    DEPTH: Scalar, span of the polygon along z axis
    EPS_IN: Scalar or Material class, epsilon of the polygon material
    EPS_OUT: Scalar or Material class, epsilon of the material outside the polygon
    EDGE_PRECISION: Positive integer, the number of quadrature points along each edge for computing the FOM gradient using shape derivative method
    """
    def __init__(self, points, z, depth, eps_in, eps_out, edge_precision) -> None:
        self.points = points
        self.z = float(z)
        self.depth = float(depth)
        self.eps_in = eps_in if isinstance(eps_in, Material) else Material(eps_in)
        self.eps_out = eps_out if isinstance(eps_out, Material) else Material(eps_out)
        self.edge_precision = int(edge_precision)
        
        if self.depth <= 0:
            raise UserWarning('polygon depth must be positive')
        if self.edge_precision <= 0:
            raise UserWarning('edge precision should be a positive integer')
        
        self.make_edge()
        self.hash = random.getrandbits(64)
        pass

    def make_edge(self):
        """Creat all the edge objects"""
        edges = []

        for i, point in enumerate(self.points):
            edges.append(Edge(self.points[i-1], point, self.eps_in, self.eps_out, self.z, self.depth))
        self.edges = edges 
        
    def get_current_params(self):
        """Get the current points coordinate aligned in 1 dimension"""
        return np.reshape(self.points, (-1)).copy()

    def add_geo(self, sim, params, only_update):
        """
        This funciton is used to add the geometry into the Lumerical file

        ---INPUTS---
        PARAMS:  1 dimensional array or 2-D array with shape (num, 2) or None. It is the   
            position of the changed vertices aligned in 1/2 dimension.
        """

        sim.fdtd.switchtolayout()
        if params is None:
            points = self.points
        else:
            points = np.reshape(params, (-1, 2))
        poly_name = "polygon_{}".format(self.hash)

        if not only_update:
            sim.fdtd.addpoly()
            sim.fdtd.set('name', poly_name)
        sim.fdtd.set('x', 0)
        sim.fdtd.set('y', 0)
        sim.fdtd.set('z', self.z)
        sim.fdtd.set('z span', self.depth)
        sim.fdtd.set('vertices', points)
        self.eps_in.set_script(sim, poly_name)

    def update_geometry(self, params):
        """
        This function is used to change and record the vertices of the polygon in the optimization, notice that 'z' and 'depth' is assumed to be unchanged in our program
        ---INPUTS---
        PARAMS: 1 dimensional array. It is the position of the changed vertices aligned in 1 
            dimension.
        """

        self.points = np.reshape(params, (-1, 2))

    def plot(self, ax):
        """
        This fucntion is used to plot the geometry of the polygon in the given axe
        ---INPUT---
        AX: Axe object. The axe that the geometry will be plotted on
        """
        points = self.points.copy()
        points = np.reshape(points, (-1, 2))
        x_p = points[:, -1] * 1e6
        y_p = points[:, -1] * 1e6
        ax.clear()
        ax.plot(x_p, y_p)
        ax.set_title('Geometry')
        ax.set_ylim(min(y_p), max(y_p))
        ax.set_xlim(min(x_p), max(x_p))
        ax.xlabel('x (um)')
        ax.ylable('y (um)')
        return True

class FunctionDefinedPolygon(Polygon):
    """
    This class is a subclass of 'Polygon class', the difference is that the vertices are given by an user-defined function, which when given a set of parameters will return a numpy array with the form [(x0, y0), ..., (xn, yn)], notice that the vertices should in the counter-clockwise direction

    ---INPUTS---
    FUNC:   Callable function, it can recieve the parameters and return array that defines the
        polygon's vertices.
    START_PARAMS:   One dimensional array, it is the initial parameters of the function
    BOUNDS: Array with dimension (num, 2), with the first column representing the minimum and 
        second column representing the maximum, it shows the bounds for the parameters
    DX:     A scalar, it is the step size of parameters when using differentiation to compute
        the gradients
    OTHERS: The same as they are defined in class 'Polygon'      
    """
    def __init__(self, func, start_params, bounds, dx, z, depth, eps_in, eps_out, edge_precision) -> None:
        self.func = func
        self.current_params = np.array(start_params).flatten()
        self.bounds = np.array(bounds)
        self.dx = float(dx)
        points = np.array(func(start_params))
        self.hist_params = []
        self.hist_params.append(self.current_params)
        super().__init__(points, z, depth, eps_in, eps_out, edge_precision)

        # Test the shape of some arguments
        assert self.bounds.shape[0] == self.current_params.size and self.bounds.shape[1] == 2
        for bound in self.bounds:
            if bound[1] - bound[0] <= 0:
                raise UserWarning('The range of "bound" should be positive')
        if dx <= 0:
            raise UserWarning('"dx" should be positive')

    def get_current_params(self):
        return self.current_params.copy()

    def add_geo(self, sim, params, only_update):
        """
        This function is used to set the geometry object in Lumerical
        ---INPUTS---
        PARAMS: 1 dimensional array. It is the parameters of the user defined function
        """
        if params is None:
            points = self.points
        else:
            points = self.func(params)
        super().add_geo(sim, points, only_update)

    def update_geometry(self, params):
        """
        This function is used to change and record the parameters in the optimization, notice that 'z' and 'depth' is assumed to be unchanged in our program
        ---INPUTS---
        PARAMS: 1 dimensional array. It is the changed parameters for the user defined function
        """
        self.current_params = params
        self.points = self.func(params)
        self.hist_params.append(params)