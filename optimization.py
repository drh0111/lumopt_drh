import os
import inspect
import shutil
import numpy as np

from Utilities.base_script import BaseScript
from Utilities.wavelength import Wavelength
from Utilities.plotter import Plotter
from Utilities.simulation import Simulation
from Utilities.gradients import GradientFields
from Figures_of_merit.modematch import ModeMatch
from Lumerical_methods.lumerical_scripts import get_field

class SuperOptimization():
    pass

class Optimization(SuperOptimization):
    """
    This class is the pivot of all the other classes, it stands as the interface between user and program. It controls all the parts (classes) to perform the commands from user (steps needed to perform the optimization)

    ---INPUTS---
    BASE_SCRIPT:    BaseScript class. It is the proxy object that help call the fdtd simulation 
        in an appropriate way.
    WAVELENGTH:     Wavelength class. It represent the wavelength range that is under interest 
        and will be set for the simulation.
    FOM:    Modematch class. It is used to calculate the figure of merit and gradient stuff.
    GEOMETRY:       Geometry class. It is used to update the geometry of the designed object.
    OPTIMIZER:      Optimizer class. It is used to accept the fom and jac, then calculate the 
        updated parameters for the designed object.
    USE_VAR_FDTD:   Bool class. It is the flag showing whether use VarFDTD session or FDTD.
    HIDE_FDTD_CAD:  Bool class. It is the flag showing whether run FDTD CAD in background or 
        not.
    DIR_GRAD:   Bool class. It is the flag showing whether use the gradients calculated from 
        FDTD directly or not.
    PLOT_HIST:  Bool class. It is the flag showing whether plot the history of parameters and 
        gradients or not.
    STORE_ALL:  Bool class. It is the flag showing whether store all the files during iteration 
        or not
    ---OTHER VARIABLES---
    FORWARD_FIELDS:
    ADJOINT_FIELDS:
    """
    def __init__(self, base_script, fom, geometry, optimizer, wavelength, use_var_fdtd = False,  hide_fdtd_cad = False, dir_grad = True, plot_hist = True, store_all = True) -> None:
        self.base_script = base_script if isinstance(base_script, BaseScript) else BaseScript(base_script)
        self.wavelength = wavelength if isinstance(wavelength, Wavelength) else Wavelength(wavelength)
        self.fom = fom
        self.geometry = geometry
        self.optimizer = optimizer
        self.use_var_fdtd = bool(use_var_fdtd)
        self.hide_fdtd_cad = bool(hide_fdtd_cad)
        self.dir_grad = bool(dir_grad)
        self.plot_hist = bool(plot_hist)
        self.store_all = bool(store_all)
        self.plotter = None # Initialize it when we know the number of parameters
        self.unfold_symmetry = geometry.unfold_symmetry
        self.hist_fom = []
        self.hist_grad = []
        self.hist_params = []

        if self.dir_grad:
            print('Use the gradient calculated from FDTD directly')

        # Set a new folder for the files in the optimization
        frame = inspect.stack()[1]
        calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        Optimization.go_to_new_opts_folder(calling_file_name, self.base_script)
        self.working_dir = os.getcwd()

    def __del__(self):
        Optimization.go_out_of_opt_dir

    def init_plotter(self):
        """This method is used to initialize the plot function"""
        if self.plotter == None:
            self.plotter = Plotter(movie = False, plot_hist = self.plotter_hist)
        return self.plotter

    def initialize(self):
        """
        Operate like a conductor, interconnect all the initialized information between different components in optimization, and perform all the steps that need to be carried out only once at the beginning of the optimization.
        """
        self.sim = Simulation(self.working_dir, self.use_var_fdtd, self.hide_fdtd_cad)

        # Set some trivial properties of FDTD
        self.base_script(self.sim.fdtd)
        Optimization.set_global_wavelength(self.sim.fdtd, self.wavelength)
        Optimization.set_source_wavelength(self.sim, 'source', self.fom.mul_freq_src, len(self.wavelength))
        self.sim.fdtd.setnamed('opt_fields', 'override global monitor settings', False)
        self.sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'none')
        Optimization.add_index_monitor(self.sim, 'opt_fields')
        Optimization.set_use_legacy_conformal_interface_detection(self.sim, self.dir_grad)

        # Initialize the properties of Optimizer, Geometry and (fom, jac)
        start_params = self.geometry.get_current_params()
        self.geometry.add_geo(self.sim, start_params, only_update = False)

        callable_fom = self.callable_fom
        callable_jac = self.callable_jac
        bounds = self.geometry.bounds

        def plotting_fun():
            """it will be used as PLOTTING_FUN to initialize the optimizer"""
            self.plotter.update(self)

            # This part is not understood yet
            if hasattr(self.geometry, 'to_file'):
                self.geometry.to_file('parameters_{}.npz'.format(self.optimizer.iter))

            with open('convergence_report.txt', 'a') as f:
                f.write('{}, {}'.format(self.optimizer.iter, self.optimizer.fom_hist[-1]))
                # This part is not understood yet
                if hasattr(self.geometry, 'write_status'):
                    self.geometry.write_status(f)
                f.write('\n')

            self.fom.initialize(self.sim)
            self.optimizer.initialize(start_params, callable_fom, callable_jac, bounds, plotting_fun)

    def run(self):
        """
        After the initialization (__init__), this final function that will be directly called by the user to run the overall optimization
        """
        self.initialize()
        self.init_plotter()

        if self.plotter.movie:
            # This part hasn't been figured out yet
            with self.plotter.writer.saving(self.plotter.fig, 'optimization.mp4', 100):
                self.optimizer.run()
        else:
            self.optimizer.run()
        
        final_fom = np.abs(self.hist_fom[-1])
        final_params = self.hist_params[-1]
        print("FINAL FOM = {}".format(final_fom))
        print("FINAL PARAMETERS = {}".format(final_params))
        return final_fom, final_params

    def make_forward_sim(self, params):
        """
        This function is used to set all the settings before running forward solver
        ---INPUT---
        PARAMS: 1-D array. The parameters of polygon (depends on its geometry type).
        """
        self.sim.fdtd.switchtolayout()
        self.geometry.update_geometry(params)
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        self.sim.fdtd.setnamed('source', 'enabled', True) # Used to set forward source
        self.fom.make_forward_sim(self.sim) # Used to set adjoint source

    def run_forward_solver(self, params):
        """
        This function is used to set and run the forward simulation and then compute the figures 
        of merit.
        ---INPUT---
        PARAMS: 1-D array. The parameters of polygon (depends on its geometry type).
        """
        print('Running forward simulation')
        self.make_forward_sim(params)
        iter = self.optimizer.iter if self.store_all else 0
        self.sim.run('Forward', iter)

        # Collect the data and calculate the fom
        get_eps = True
        get_D = not self.dir_grad
        nointerpolation = not self.geometry.use_interpolation()
        self.forward_fields = get_field(self.sim.fdtd, 
                                        monitor_name = 'opt_fields', # named in base script
                                         field_result_name = 'forward_fields',
                                         get_eps = get_eps,
                                         get_D = get_D,
                                         get_H = False,
                                         noninterpolation = nointerpolation,
                                         unfold_symmetry = self.unfold_symmetry)

        fom = self.fom.get_fom(self.sim)
        if self.store_all:
            self.sim.save()
        self.hist_fom.append(fom)
        print('FOM = {}'.format(fom))
        return fom
    
    def make_adjoint_sim(self, params):
        """
        This function is used to set all the settings before running adjoint solver
        ---INPUT---
        PARAMS: 1-D array. The parameters of polygon (depends on its geometry type).
        """
        self.sim.fdtd.switchtolayout()
        assert np.allclose(params, self.geometry.get_current_params), 'discrepancy of parameters, run forward simulation (update_params) first'
        self.geometry.add_geo(self.sim, params = None, only_update = True)
        self.sim.fdtd.setnamed('source', 'enabled', False)
        self.fom.make_adjoint_sim(self.sim)

    def run_adjoint_solves(self, params):
        """
        This function is used to set and run the adjoint simulation and then extract the adjoint field
        ---INPUT---
        PARAMS: 1-D array. The parameters of polygon (depends on its geometry type).
        """
        has_forward_field = hasattr(self, 'forward_fields') and hasattr(self.forward_fields, 'E')
        if not has_forward_field:
            self.run_forward_solver(params)
        
        print('Running adjoint simulation')
        self.make_adjoint_sim(params)
        iter = self.optimizer.iter if self.store_all else 0
        self.sim.run(name = 'adjoint', num = iter)

        get_eps = not self.dir_grad
        get_D = not self.dir_grad
        nointerpolation = not self.geometry.use_interpolation()

        self.adjoint_fields = get_field(self.sim.fdtd,
                                        monitor_name = 'opt_fields',
                                        field_result_name = 'adjoint_fields',
                                        get_eps = get_eps,
                                        get_D = get_D,
                                        get_H = False,
                                        noninterpolation = nointerpolation,
                                        unfold_symmetry = self.geometry.unfold_symmetry)
        self.adjoint_fields.scaling_factor = self.fom.get_adjoint_field_scaling(self.sim)
        self.adjoint_fields.scale(3, self.adjoint_fields.scaling_factor)

        if self.store_all:
            self.sim.save()

    def calculate_gradients(self):
        """
        After getting the forward fields and adjoint fields, this function is used to calculate the gradients of fom with respect to parameters. There are two method to calculate the gradients:
            1) dir_grad == True. Use direct method (perturb the parameters) to get the permittivity derivative calculated directly from meshing
            2) dir_grad == False. Use the shape derivative approximation described in Owen Miller's thesis
        """
        print('Calculating gradients')
        fdtd = self.sim.fdtd
        self.gradient_fields = GradientFields(self.forward_fields, self.adjoint_fields)
        self.sim.fdtd.switchtolayout()
        if self.dir_grad:
            self.geometry.d_eps_on_cad(self.sim)
            fom_partial_derivs_vs_wl = GradientFields.spatial_gradient_integral_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.adjoint_fields.scaling_factor)
            self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl.transpose(), self.forward_fields.wl) # tranpose is for the alignment of input
        else:
            if hasattr(self.geometry, 'calculate_gradients_on_cad'):
                # This case is not understood yet
                fom_partial_derivs_vs_wl = self.geometry.calculate_gradients_on_cad(self.sim, 'forward_fields', 'adjoint_fields', self.adjoint_fields.scaling_factor)
                self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, self.forward_fields.wl)
            else:
                fom_partial_derivs_vs_wl = self.geometry.calculate_gradients(self.gradient_fields)
                self.gradients = self.fom.fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl, self.forward_fields.wl)
        return self.gradients
                
    def callable_fom(self, params):
        """
        This function is used to calculate the figures of merit
        ---INPUT---
        PARAMS: Array class, it is the parameters needed to be optimized
        ---OUTPUT---
        RETURN: scalar, it is the figures of merit
        """
        return self.run_forward_solver(params)

    def callable_jac(self, params):
        """
        This function is used to calculate the jacobian (gradient) of the figure of merits
        ---INPUTS---
        PARAMS: Array class, it is the parameters needed to be optimized
        ---OUTPUT---
        RETURN: Array, the jacobian gradients of the figure of merits
        """
        self.run_adjoint_solves(params)
        return self.calculate_gradients()

    @staticmethod
    def go_to_new_opts_folder(calling_file_name, base_script):
        """
        This method will create a new folder named opt_xx in the current working directory, it will store all the files during simulation and the copy of base_script in the new folder
        """
        working_dir = os.path.dirname(calling_file_name) if os.path.isfile(calling_file_name) else os.getcwd()
        working_dir_split = os.path.split(working_dir)
        if working_dir_split[1].startswith('opt_'):
            working_dir = working_dir_split[0]

        working_dir_entries = os.listdir(working_dir)
        used_num = [int(entry.split('_')[-1]) for entry in working_dir_entries if entry.startswith('opt_')]
        used_num.append(-1)
        new_opt_dir = os.path.join(working_dir, 'opt_{}'.format(max(used_num) + 1))
        os.makedirs(new_opt_dir)
        os.chdir(new_opt_dir)

        # copy the base script and python file into the new directory
        if os.path.isfile(calling_file_name):
            shutil.copy(calling_file_name, new_opt_dir)
        if hasattr(base_script, 'project'):
            shutil.copy(base_script.project, new_opt_dir)
        elif hasattr(base_script, 'script'):
            with open('script_file.lsf', 'a') as file:
                file.write(base_script.script.replace(';', ';\n'))
            

    @staticmethod
    def go_out_of_opt_dir():
        cwd_split = os.path.split(os.path.abspath(os.getcwd()))
        if cwd_split[1].startswith('opt_'):
            os.chdir(cwd_split[0])

    @staticmethod
    def set_global_wavelength(sim, wavelength):
        """
        Set the global wavelength of FDTD simulation
        ---INPUTS---
        SIM: Simulation class.
        WAVELENGTH: Wavelength class.
        """
        sim.fdtd.setglobalsource('set wavelength', True)
        sim.fdtd.setglobalsource('wavelength start', wavelength.min())
        sim.fdtd.setglobalsource('wavelength stop', wavelength.max())

        sim.fdtd.setglobalmonitor('use source limits', True)
        sim.fdtd.setglobalmonitor('use linear wavelength spacing', True)
        sim.fdtd.setglobalmonitor('frequency points', len(wavelength))
        
    @staticmethod
    def set_source_wavelength(sim, source_name, mul_freq_src, freq_pts):
        """
        This function is used to set the wavelength of source object
        ---INPUTS---
        SIM:    Simulation class.
        SOURCE_NAME:    String class. 
        MUL_FREQ_SRC: Bool class. Whether set multifrequency beam calculation or not
        """
        if sim.fdtd.getnamednumber(source_name) != 1:
            raise UserWarning('there should be one and only one source named {} in the simulation'.format(source_name))
        if sim.fdtd.getnamed(source_name, 'override global source setting'):
            print('The wavelength setting of the source will bw superseded by the global settings')
        sim.fdtd.setnamed(source_name, 'override global source settings', False)

        sim.fdtd.select(source_name)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifreqency mode calculation', mul_freq_src)
            if mul_freq_src:
                sim.fdtd.setnamed(source_name, 'frequency points', freq_pts)
        elif sim.fdtd.haveproperty('multifreqency beam calculation'):
            sim.fdtd.setnamed(source_name, 'multifreqency beam calculation', mul_freq_src)
            if mul_freq_src:
                sim.fdtd.setnamed(source_name, 'number of frequency points', freq_pts)            
        elif sim.fdtd.haveproperty('frequency dependent profile'):
            sim.fdtd.setnamed(source_name, 'frequency dependent profile', mul_freq_src)
            if mul_freq_src:
                sim.fdtd.setnamed(source_name, 'number of field profile samples', freq_pts)
        else:
            raise UserWarning('Cannot set multifrequency mode calculation')

    @staticmethod
    def add_index_monitor(sim, mon_name):
        """
        This function is used to add the dielectric index monitor in the simulation, be aware that 'mon_name' is not the monitor for the calculation of fom, it is monitor based on which we will calculate gradients, and is usually named 'opt_fields'
        ---INPUTS---
        MON_NAME: String class. It is the name of monitor based on which we will set the 
            properties of index monitor, like monitor type...
        """

        # Check the existence of the 'mon_name', add the monitor
        if sim.fdtd.getnamednumber(mon_name) != 1:
            raise UserWarning('there should be one and only one monitor named {} in the simulation'.format(mon_name))
        index_mon_name = mon_name + '_index'
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addindex()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addeffectiveindex()
        else:
            raise UserWarning('There shoule be FDTD or varFDTD object')

        # Set the properties of the index monitor
        sim.fdtd.set('name', index_mon_name)
        sim.fdtd.setnamed(index_mon_name, 'override global monitor settings', True)
        sim.fdtd.setnamed(index_mon_name, 'frequency points', 1)
        sim.fdtd.setnamed(index_mon_name, 'record conformal mesh when possible', True)
        sim.fdtd.setnamed(index_mon_name, 'spatial interpolation', 'none')
        mon_type = sim.fdtd.getnamed(mon_name, 'monitor type')
        props = ['monitor type']
        props.extend(Optimization.cross_section_monitor_props(mon_type))
        for prop in props:
            value = sim.fdtd.getnamed(mon_name, prop)
            sim.fdtd.setnamed(index_mon_name, prop, value)

    @staticmethod
    def cross_section_monitor_props(mon_type):
        """
        This function is used to find the properties that need to be set for the given monitor type.
        ---OUTPUT---
        GEO_PROPS: List class. Contain the strings that show the properties need to be set under 
            the given monitor type.
        """
        geo_props = ['x', 'y', 'z']
        if mon_type == '3D':
            geo_props.extend(['x span', 'y span', 'z span'])
        elif mon_type == '2D X-normal':
            geo_props.extend(['y span', 'z span'])
        elif mon_type == '2D Y-normal':
            geo_props.extend(['x span', 'z span'])
        elif mon_type == '2D Z-normal':
            geo_props.extend(['x span', 'y span'])
        elif mon_type == 'Linear X':
            geo_props.append('x span')
        elif mon_type == 'Linear Y':
            geo_props.append('y span')
        elif mon_type == 'Linear Z':
            geo_props.append('z span')
        else:
            raise UserWarning('invalid monitor type')
        return geo_props

    @staticmethod
    def set_use_legacy_conformal_interface_detection(sim, flagVal):
        # The meaning of this function is unknown yet
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.select('FDTD')
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.select('varFDTD')
        else:
            raise UserWarning('There shoule be FDTD or varFDTD object')
        
        if bool(sim.fdtd.haveproperty('use legacy conformal interface detection')):
            sim.fdtd.set('use legacy conformal interface detection', flagVal)
            sim.fdtd.set('conformal meshing refinement', 51)
            sim.fdtd.set('mesh tolerance', 1.0/1.134e14)
        else:
            raise UserWarning('need more recent version of FDTD tor the permittivity derivatives will not be accurate')

