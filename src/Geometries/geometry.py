import sys
import numpy as np
import lumapi

class Geometry():
    """
    It is a more abstract level of geometry, which is consist of many explicit geometries like polygon... It can be understood as a geometry collector, whose method is actually try to call each member's same name method in an appropriate way.
    """
    unfold_symmetry = True # We will unfold the symmetry of the monitor by default

    def use_interpolation(self):
        """Given the flag showing whether use interpolation or not"""
        return False

    def __init__(self, geometries, operation) -> None:
        """
        ---INPUTS---
        GEOMETRIES: List of geometry class.
        OPERATION:  String, 'mul' or ''add
        ---OTHER VARIABLE---
        BOUNDS: 2-D array with shape (num, 2), the bounds of all parameters
        """
        self.geometries = geometries
        self.operation = operation
        if operation == 'add':
            self.bounds = geometries[0].bounds
        elif operation == 'mul':
            self.bounds = np.concatenate([np.array(geometries[0].bounds), np.array(geometries[1].bounds)])
        else:
            raise UserWarning('operation variable can only be "add" or "mul"')
        self.dx = max([geometries[0].dx, geometries[1].dx])

    def __add__(self, other):
        """Two geometry with independent parameters"""
        geometries = [self, other]
        return Geometry(geometries, 'add')

    def __mul__(self, other):
        """Two geometry with the same parameters"""
        geometries = [self, other]
        return Geometry(geometries, 'mul')
    
    def __len__(self):
        """This function is used to get the lens of parameters of the geometry object"""
        # Because for the explicit 
        return len(self.get_current_params())

    def update_geometry(self, params):
        """
        This function is used to update geometry for every member in geometries
        ---INPUTS---
        PARAMS: 1-D array.
            It is the parameters of geometries (if it's polygon, then should align it in 1-D) aligned in 1-D.
        """
        if self.operation == 'mul':
            self.geometries[0].update_geometry(params)
            self.geometries[1].update_geometry(params)
        elif self.operation == 'add':
            num = len(self.geometries[0].get_current_params)
            self.geometries[0].update_geometry(params[ : num])
            self.geometries[1].update_geometry(params[ : num])
        else:
            raise UserWarning('operation variable can only be "add" or "mul"')

    def add_geo(self, sim, params, only_update):
        """
        This function is used to add geometry for every member in geometries
        ---INPUTS---
        PARAMS: 1-D array.
            The parameter of each geometry aligned in 1-D array
        """
        if self.operation == 'mul':
            self.geometries[0].add_geo(sim, params, only_update)
            self.geometries[1].add_geo(sim, params, only_update)
        elif self.operation == 'add':
            num = len(self.geometries[0]) # the point divide parameters of geo_1 and geo_2
            self.geometries[0].add_geo(sim, params[ : num], only_update)
            self.geometries[1].add_geo(sim, params[num : ], only_update)
        else:
            raise UserWarning('operation variable can only be "add" or "mul"')

    def get_current_params(self):
        """
        This function will return the parameters for all the geometry aligned in 1-D array
        """
        if self.operation == 'mul':
            return np.array(self.geometries[0].get_current_params())
        elif self.operation == 'add':
            return np.concatenate([np.array(self.geometries[0].get_current_params()), np.array(self.geometries[1].get_current_params())])

    def d_eps_on_cad(self, sim, monitor_name = 'opt_fields'):
        """
        This function will be used to calculate gradients for dir_grad == True case, it will calculate the difference of eps after perturbing every parameter respectively, which combined with forward and adjoint fields can give out the gradients. be aware that the data will only be stored in Lumerical
        """
        Geometry.get_eps_from_index_monitor(sim.fdtd, 'original_eps_data', monitor_name)
        current_params = self.get_current_params()
        sim.fdtd.eval("d_epses = cell({});".format(len(current_params)))
        lumapi.putDouble(sim.fdtd.handle, 'dx', self.dx)
        print('get d eps under dx =' + str(self.dx))
        sim.fdtd.redrawoff()
        for i, param in enumerate(current_params):
            d_params = current_params.copy()
            d_params[i] = param + self.dx
            self.add_geo(sim, d_params, only_update = True)
            Geometry.get_eps_from_index_monitor(sim.fdtd, 'current_eps_data')
            sim.fdtd.eval("d_epses{" + str(i+1) + "} = (current_eps_data - original_eps_data) / dx;")
            sys.stdout.write('.'); sys.stdout.flush() # Problem: ',' or ';', personal: used to record the process
        sim.fdtd.eval("clear(original_eps_data, current_eps_data, dx);")
        print('') # used to make a newline
        sim.fdtd.redrawon()
        self.add_geo(sim, current_params, only_update = True) # recover the original geometry

    @staticmethod
    def get_eps_from_index_monitor(fdtd, eps_result_name, monitor_name = 'opt_fields'):
        """
        This function is used to get the dielectric constant of the geometry's cross section through the index monitor, be aware that the data is store in Lumerical instead of python
        ---INPUTS---
        FDTD: FDTD object.
            Based on which we colloect the data
        EPS_RESULT_NAME:    String.
            The name of matrix variable in Lumerical that will store eps_data, with size (x, y, z, vec_eps)
        MONITOR_NAME:   String.
            The name of monitor based on which eps_monitor is constructed
        """
        index_monitor_name = monitor_name + '_index'
        assert fdtd.getnamednumber(monitor_name) == 1, 'Don not have the desired field monitor'
        assert fdtd.getnamednumber(index_monitor_name) == 1, 'Don not have the desired index monitor'
        fdtd.eval("{0}_data_set = getresult('{0}', 'index');".format(index_monitor_name) + 
                    "{0} = matrix(length({1}.x), length({1}.y), length({1}.z), length({1}.f), 3);".format(eps_result_name, index_monitor_name + '_data_set') + 
                    "{0}(:, :, :, :, 1) = {1}_data_set.index_x ^ 2;".format(eps_result_name, index_monitor_name) +
                    "{0}(:, :, :, :, 2) = {1}_data_set.index_y ^ 2;".format(eps_result_name, index_monitor_name) + 
                    "{0}(:, :, :, :, 3) = {1}_data_set.index_z ^ 2;".format(eps_result_name, index_monitor_name) + 
                    "clear({0}_data_set);".format(index_monitor_name))

