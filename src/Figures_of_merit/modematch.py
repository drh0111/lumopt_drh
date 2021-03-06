from cmath import cos
import numpy as np
import scipy as sp
import scipy.constants

from Utilities.wavelength import Wavelength
from Polarization.extract_polar import extract_polar

def is_int(x):
    """This function is used to test whether the value x is or can be transfered into integer"""
    try:
        a = int(x)
        return True
    except ValueError:
        return False

class ModeMatch:
    """
    This class is used to calculate the figures_of_merit of the optimization by overlap integral of the simulation fields recorded by the field monitor and the desired eigenmodes. The integral is completed by the mode expansion monitor, and the results are recorded in the T-forward variable, the detail of which can be found in the following page:
            https://kb.lumerical.com/ref_sim_obj_using_mode_expansion_monitors.html
    """
    def __init__(self, monitor_name, mode_num, direction, mul_freq_src = False, target_T_foward = lambda wl: np.ones(wl.size), norm_p = 1):
        """
        This function is used to assign the value of class variables
        ---INPUTS---
        MONITOR_NAME:   String. The name of monitor from which we will get the information  
            of fields
        MODE_NUMB:      positive integer or some specific string. The index of eigenmode or 
            string that Lumerical can recognize which we will use mode expansion monitor to calculate, and then used in the mode source for adjoint simulation
        DIRECTION:      String. The direction of the injected source, 'Forward' or 'Backward'
        MUL_FREQ_SRC:   False or True. Represent whether the multifrequncy mode for source 
            in Lumerical is on or off.
        TARGET_T_FORWARD:   Function, with the array of wavelength as input, it will return 
            an array representing the target T-forward values.
        NORM_P:  Scalar. The exponent we use to generate the figures of merit
        ---OTHER VARIABLES---
        WAVELENGTH: Array, represent the sampling points of wavelength
        PHASE_PREFACTORS:    1-D Array,
        """
        if not monitor_name:
            raise UserWarning('empty monitor name')
        self.monitor_name = str(monitor_name)
        self.exp_mon_name = self.monitor_name + '_mode_exp'
        self.mode_src_name = self.monitor_name + '_mode_src'

        if is_int(mode_num):
            self.mode_num = int(mode_num)
            if self.mode_num < 0:
                raise UserWarning('mode_num cannot be negative')
        else:
            self.mode_num = mode_num

        assert direction == 'Forward' or direction == 'Backward'
        self.direction  = direction
        self.mul_freq_src = mul_freq_src

        # Test the correctness of 'target_T_foward' function
        test_result = target_T_foward(np.linspace(0.1e-6, 10e-6, 1000))
        if test_result.size != 1000:
            raise UserWarning('target_T_foward should return as many values as the input wavelngth')
        elif np.any(test_result.min() < 0) or np.any(test_result.max() > 1):
            raise UserWarning('target_T_foward should return values between 0 and 1')
        else:
            self.targe_T_forward = target_T_foward

        if norm_p < 0:
            raise UserWarning('norm_p should be positive')
        else:
            self.norm_p = norm_p

    def initialize(self, sim):
        """
        This function is used to initialize and check the setting of simulation for the calculation of figures_of_merit, while __init__ function is mainly focused on value assignment
        """

        self.check_monitor_alignment(sim)
        ModeMatch.add_mode_exp_monitor(sim, self.monitor_name, self.exp_mon_name, self.mode_num)
        adj_inj_dir = 'Backward' if self.direction == 'Forward' else 'Forward'
        ModeMatch.add_mode_src(sim, self.monitor_name, self.mode_src_name, adj_inj_dir, self.mode_num, self.mul_freq_src)

    def get_fom(self, sim):
        """
        This function is used to calculate the fom after the forward simulation
        """
        trans_coeff = ModeMatch.get_transmission_coefficient(sim, self.direction, self.monitor_name, self.exp_mon_name)
        self.wavelength = ModeMatch.get_wavelength(sim)
        source_power = ModeMatch.get_source_power(sim, self.wavelength)
        self.T_fwd_vs_wl = np.real(trans_coeff * trans_coeff.conj() / source_power)
        self.phase_prefactors = trans_coeff / 4.0 / source_power # Problem: meaning
        fom = ModeMatch.fom_wavelength_integral(self.T_fwd_vs_wl, self.wavelength, self.targe_T_forward, self.norm_p)
        return fom

    def check_monitor_alignment(self, sim, tol = 1e-9):
        """
        This function is used to check the field monitor used to calculate the figures_of_merit, in particular to ensure that monitor is set close enough to the simulation grids

        ---INPUTS---
        TOL:    Scalar. It represents the largest acceptable distance between monitor and 
            simulation grids.
        """

        if sim.fdtd.getnamednumber(self.monitor_name) != 1:
            raise UserWarning('there shouold be only one field monitor for fom')
        
        # Get the orientation of the monitor
        monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')
        if monitor_type == '2D X-normal':
            orientation = 'x'
        elif monitor_type == '2D Y-normal':
            orientation = 'y'
        elif monitor_type == '2D Z-normal':
            orientation = 'z'
        else:
            raise UserWarning('invalid monitor type')

        # check the alignment of field monitor is close enough to the simulation grids
        mon_pos = sim.fdtd.getnamed(self.monitor_name, orientation)
        if sim.fdtd.getnamednumber('FDTD') == 1:
            grids = sim.fdtd.getresult('FDTD', orientation)
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            grids = sim.fdtd.getresult('varFDTD', orientation)
        else:
            raise UserWarning('no FDTD or varFDTD slover can be found')

        if min(grids - mon_pos) > tol:
            raise UserWarning('distance between monitor and simulation grids is out of tolerance, it will result in phase error and inaccurate gradients')

    def get_adjoint_field_scaling(self, sim):
        """
        Due to the set of fom and thus the form of E_adjoint 
        (refer to  https://www.osapublishing.org/oe/abstract.cfm?uri=oe-21-18-21693)
        and the normalization principle of modal fields
        (refer to https://kb.lumerical.com/ref_sim_obj_using_mode_expansion_monitors.html)
        there should be scaling term for the adjoint fields, which will be returned by this function
        ---INPUT---
        SIM:    Fdtd object, used to get source power Problem: src_pwr the same as forward one?
        """
        omega = 2 * np.pi * scipy.constants.speed_of_light / self.wavelength
        adjoint_source_power = ModeMatch.get_source_power(sim, self.wavelength)

        assert hasattr(self, 'phase_prefactors'), 'Don not have phase_prefactors attribute'
        #scaling_factor = np.conj(self.phase_prefactors) * omega * 1j / np.sqrt(adjoint_source_power)
        scaling_factor = np.conj(self.phase_prefactors) * omega / np.sqrt(adjoint_source_power)
        return scaling_factor

    @staticmethod
    def add_mode_exp_monitor(sim, monitor_name, exp_mon_name, mode_num):
        """
        This function is used to set the mode expansion monitor for the calculation of fom
        """
        
        # Check the field monitor
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning('there shouold be only one field monitor for fom')
        sim.fdtd.setnamed(monitor_name, 'override global monitor settings', False)

        # Set a mode expansion monitor for the existing field monitor
        if sim.fdtd.getnamednumber(exp_mon_name) != 0:
            raise UserWarning('a mode expansion monitor already exist')
        else:
            sim.fdtd.addmodeexpansion()
            sim.fdtd.set('name', exp_mon_name)
            sim.fdtd.setexpansion(exp_mon_name, monitor_name)
            sim.fdtd.setnamed(exp_mon_name, 'auto update before analysis', True)
            sim.fdtd.setnamed(exp_mon_name, 'override global monitor settings', False)

            # Synchronize the properties of mode expansion monitor
            props = ['monitor type']
            monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
            geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
            props.extend(geo_props)

            for prop in props:
                prop_val = sim.fdtd.getnamed(monitor_name, prop)
                sim.fdtd.setnamed(exp_mon_name, prop, prop_val)

            # Select the desired mode
            if is_int(mode_num):
                sim.fdtd.setnamed(exp_mon_name, 'mode selection', 'user select')
                sim.fdtd.updatemodes(mode_num)
            else:
                sim.fdtd.setnamed(exp_mon_name, 'mode selection', mode_num)
                sim.fdtd.updatemodes()

            # elif isinstance(mode_num, list):
            #     # transform list into string that can be recognized by Lumerical
            #     mode = ",".join([str(x) for x in mode_num])
            #     sim.fdtd.setnamed(exp_mon_name, 'mdoe selection', 'user select')
            #     sim.fdtd.updatemodes(mode)
    
    @staticmethod
    def cross_section_monitor_props(monitor_type):
        """
        Given the field monitor's type, this function will return a list containing the properties of the monitor that mode expansion monitor should synchronize
        """
        props = ['x', 'y', 'z']
        normal = ''
        if monitor_type == '2D X-normal':
            props.extend(['y span', 'z span'])
            normal = 'x'
        elif monitor_type == '2D Y-normal':
            props.extend(['x span', 'z span'])
            normal = 'y'
        elif monitor_type == '2D Z-normal':
            props.extend(['x span', 'y span'])
            normal = 'z'
        else:
            raise UserWarning('invalid monitor type for field monitor')

        return props, normal

    @staticmethod
    def add_mode_src(sim, monitor_name, mode_src_name, direction, mode_num, mul_freq_src):
        """
        Add and set the mode source in Lumercal for the simulation of adjoint problem
        """

        # add the mode source
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addmode()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addmodesource()
        else:
            raise UserWarning('cannot find FDTD or varFDTD object')

        sim.fdtd.set('name', mode_src_name)
        sim.fdtd.select(mode_src_name)

        # synchronize the properties of mode source
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)

        sim.fdtd.setnamed(mode_src_name, 'injection axis', normal.lower() + '-axis')
        if sim.fdtd.getnamednumber('varFDTD') == 1:
            geo_props.remove('z')
        for prop in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop)
            sim.fdtd.setnamed(mode_src_name, prop, prop_val)
        sim.fdtd.setnamed(mode_src_name, 'override global source settings', False)
        sim.fdtd.setnamed(mode_src_name, 'direction', direction)
        if is_int(mode_num):
            sim.fdtd.setnamed(mode_src_name, 'mode selection', 'user select')
            sim.fdtd.select(mode_src_name)
            sim.fdtd.updatesourcemode(int(mode_num))
        else:
            sim.fdtd.setnamed(mode_src_name, 'mode selection', mode_num)
            sim.fdtd.select(mode_src_name)
            sim.fdtd.updatesourcemode()

    @staticmethod
    def get_transmission_coefficient(sim, direction, monitor_name, exp_mon_name):
        """
        This function is used to get the normalized (energy of each mode is normalized) complex transmission coefficient from the given monitor
        ---INPUTS---
        DIRECTION: String, 'Forward' or  'Backward'
            The defined transmission direction (and thus the calculated one) of the monitor.
        """
        result_name = 'expansion for ' + exp_mon_name
        if not sim.fdtd.haveresult(exp_mon_name, result_name):
            raise UserWarning('Cannot find mode expansion result')
        dataset = sim.fdtd.getresult(exp_mon_name, result_name)
        fwd_trans_coeff = dataset['a'] * np.sqrt(dataset['N'].real)
        bac_trans_coeff = dataset['b'] * np.sqrt(dataset['N'].real)
        if direction == 'Backward':
            fwd_trans_coeff = bac_trans_coeff
        return fwd_trans_coeff.flatten()

    @staticmethod
    def get_wavelength(sim):
        """
        This function will return the array according to simulation's setting
        ---OUTPUT---
        Array, represent the sampling points of wavelength
        """
        return Wavelength(sim.fdtd.getglobalsource('wavelength start'),
                            sim.fdtd.getglobalsource('wavelength stop'),
                            sim.fdtd.getglobalmonitor('frequency points')).asarray()

    @staticmethod
    def get_source_power(sim, wavelength):
        """
        This function will return the source power of the given wavelength
        ---INPUT---
        WAVELENGTH: 1-D array
            The sampling points of the expected wavelength in 'm' unit
        """
        frequency = scipy.constants.speed_of_light / wavelength
        sourcepower = sim.fdtd.sourcepower(frequency)
        return  np.asarray(sourcepower).flatten()

    @staticmethod
    def fom_wavelength_integral(T_fwd_vs_wl, wavelength, target_T_forward, norm_p):
        """
        This function is used to calculate the fom through integral
        ---INPUTS---
        T_FWD_VS_WL:    1-D array
            The calculated forward transmission coefficients
        WAVELENGTH: 1-D array
            The sampled points for the desired wavelength
        TARGER_T_FORWARD:   Callable function
            Wavelengths as input and target transmission coefficients as output
        ---OUTPUT---
        FOM:    Real number
        """
        target_T_fwd_vs_wl = target_T_forward(wavelength).flatten()
        if len(wavelength) > 1:
            wavelength_range = wavelength.max() - wavelength.min()
            assert wavelength_range > 0, 'wavelength range should be positive'
            T_fwd_integrand = np.power(np.abs(target_T_fwd_vs_wl), norm_p) / wavelength_range # used to normalize
            const_term = np.power(np.trapz(y = T_fwd_integrand, x = wavelength), 1.0 / norm_p)
            T_fwd_error = np.abs(target_T_fwd_vs_wl - T_fwd_vs_wl.flatten())
            T_fwd_error_integrand = np.power(T_fwd_error, norm_p) / wavelength_range
            error_term = np.power(np.trapz(y = T_fwd_error_integrand, x = wavelength), 1.0 / norm_p)
            fom = const_term - error_term
        else:
            fom = np.abs(target_T_fwd_vs_wl) - np.abs(T_fwd_vs_wl.flatten() - target_T_fwd_vs_wl)
        return fom.real

    def make_forward_sim(self, sim):
        """
        This function is used to set the adjoint source before running forawrd simulation
        --INPUT---
        SIM:    Simulation object.
        """

        sim.fdtd.setnamed(self.mode_src_name, 'enabled', False)

    def make_adjoint_sim(self, sim):
        """
        This function is used to set the adjoint source before running adjoint simulation
        --INPUT---
        SIM:    Simulation object.
        """

        sim.fdtd.setnamed(self.mode_src_name, 'enabled', True)

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        """
        This function is used to calculate the final FOM gradients over parameters after considering the function's form (norm_p staff) and integrating over wavelength
        ---INPUT---
        WL: 1-D array.
            The selected wavelength points for T_fwd_partial_derivs_vs_wl
        """
        assert np.allclose(wl, self.wavelength), 'wavelength pts of the desired variable does not fit'
        return ModeMatch.fom_gradient_wavelength_integral_impl(self.T_fwd_vs_wl, T_fwd_partial_derivs_vs_wl, self.targe_T_forward(wl).flatten(), self.wavelength, self.norm_p)

    @staticmethod
    def fom_gradient_wavelength_integral_impl(T_fwd_vs_wl ,T_fwd_partial_derivs_vs_wl , target_T_fwd_vs_wl, wl, norm_p):
        """
        This implicit function is used in the end to calculate the final FOM gradients over parameters after considering the function's form (norm_p staff) and integrating over wavelength
        ---INPUTS---
        T_FWD_VS_WL: 1-D array.
            This is the calculated T_fwd versus wavelength
        T_FWD_PARTIAL_DERIVS_VS_WL: 2-D array with shape (num_param, num_wl).
            This is the FOM gradients over parameters versus wavelength
        TARGET_T_FWD_VS_WL:   1-D array.
            This is the target T_fwd versus wavelength
        WL:`1-D array.
            The selected wavelength points
        NORMP_P:    Real number.
            the exponent of the final FOM function
        """
        if wl.size > 1:
            assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size, 'inconsistent wavelength points'
            wavelength_range = wl.max() - wl.min()
            T_fwd_error = T_fwd_vs_wl - target_T_fwd_vs_wl
            T_fwd_error_integrand = np.power(np.abs(T_fwd_error), norm_p) / wavelength_range
            constant_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wl), 1/norm_p - 1) # it is the multiplier constant factor
            integral_kernel = np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / wavelength_range
            # Problem: Why it will be faster? (see standard code)
            d = np.diff(wl)
            quad_weight = np.append(np.append(d[0], d[ : -1] + d[1 : ]), d[-1]) / 2
            v = constant_factor * integral_kernel * quad_weight
            T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.dot(v)
        else:
            T_fwd_partial_derivs = -1.0 * np.sign(T_fwd_vs_wl - target_T_fwd_vs_wl) * T_fwd_partial_derivs_vs_wl.flatten()
        return T_fwd_partial_derivs.flatten().real

class PolarMatch(ModeMatch):
    """
    This class is modified based on ModeMatch class. The difference is that FOM here is constructed based on the transimission of desired polarization, thus details like the calculation and the setting of adjoint source are different.
    """

    def __init__(self, monitor_name, theta, phi, pol_state, direction, mul_freq_src=False, target_T_foward=lambda wl: np.ones(wl.size), norm_p=1):
        """
        This function is used to assign the value of class variables
        ---INPUTS---
        MONITOR_NAME:   String. The name of monitor from which we will get the information  
            of fields and calculate FOM.
        THETA, PHI:    Real values. It represents the theta angle and phi angle of the incident sources that we are interested in and will calculate the transmission coefficient, in degree unit. We follow the definition in Lumerical FDTD.
        POL_STATE:  Normalized two-element complex-valued array (p_s, p_p). It represents 
            the expected polarization of transmission light, s and p polarization standard follows the one in Polaztion part.
        DIRECTION:      String. The direction of the injected source, 'Forward' or 'Backward'
            , only 'Backward' case is considered here.
        MUL_FREQ_SRC:   False or True. Represent whether the multifrequncy mode for source 
            in Lumerical is on or off.
        TARGET_T_FORWARD:   Function, with the array of wavelength as input, it will return 
            an array representing the target T-forward values (for expected polarization).
        NORM_P:  Scalar. The exponent we use to generate the figures of merit
        ---OTHER VARIABLES---
        WAVELENGTH: Array, represent the sampling points of wavelength
        PHASE_PREFACTORS:    1-D Array,
        """
        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('Empty monitor name.')
        self.adjoint_src_names = [monitor_name+'_adj_src_s', monitor_name+'_adj_src_p']
        if np.abs(np.linalg.norm(pol_state)-1) > 0.0001:
            raise UserWarning('K parallel is not normalized')
        self.pol_state = pol_state
        assert direction == 'Forward' or direction == 'Backward', 'Invalid direction'
        self.direction  = direction
        self.theta = theta
        self.phi = phi
        self.mul_freq_src = mul_freq_src

        # Test the correctness of 'target_T_foward' function
        test_result = target_T_foward(np.linspace(0.1e-6, 10e-6, 1000))
        if test_result.size != 1000:
            raise UserWarning('target_T_foward should return as many values as the input wavelngth')
        elif np.any(test_result.min() < 0) or np.any(test_result.max() > 1):
            raise UserWarning('target_T_foward should return values between 0 and 1')
        else:
            self.targe_T_forward = target_T_foward

        if norm_p < 0:
            raise UserWarning('norm_p should be positive')
        else:
            self.norm_p = norm_p

    def initialize(self, sim):
        """
        This function is used to initialize and check the setting of simulation for the calculation of figures_of_merit, while __init__ function is mainly focused on value assignment
        """

        self.check_monitor_alignment(sim)
        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'
        PolarMatch.add_plane_srcs(sim, self.monitor_name, adjoint_injection_direction, self.adjoint_src_names, -np.conjugate(self.pol_state), self.theta, self.phi) # Problem -conjugate or not
    
    @staticmethod
    def add_plane_srcs(sim, monitor_name, direction, source_names, pol_state, theta=0, phi=0):
        """
        This function will set the plane wave sources for adjoint simulation according to the construction of FOM (POL_STATE, THETA, PHI)
        ---INPUTS---
        SIM:    Simulation object.
        SOURCE_NAMES:   Two-element list
            Names of adjoint sources, [src_s, src_p]
        POL_STATE:  2-element complex-valued array
            Normalized polarization state of source we are going to add (be aware that it is different from the POL_STATE in FOM). It is in the form [s_pol, p_pol], and the definition is consistant with Polarization part of the program. The definition here is not consistent with the x, y polariztion of sources in Lumerical FDTD (polarization angle = 90, 0), so attention is needed for oblique incidence, but for normal incidence this function will work.
        """

        [s_amp, p_amp] = np.abs(pol_state)
        [s_phase, p_phase] = np.angle(pol_state, deg=True)
        PolarMatch.add_plane_src(sim, monitor_name, source_name=source_names[1], direction=direction, theta=theta, phi=phi, pol_angle=0, amp=p_amp, phase=p_phase)
        PolarMatch.add_plane_src(sim, monitor_name, source_name=source_names[0], direction=direction, theta=theta, phi=phi, pol_angle=90, amp=s_amp, phase=s_phase)
        

    @staticmethod
    def add_plane_src(sim, monitor_name, source_name, direction, theta=0, phi=0, pol_angle=0, amp=1, phase=0):
        """
        This function is used to set the desired plane wave source
        ---INPUTS---
        MONITOR_NAME:   String.
            The name of monitor from which we copy the properties for our plane wave source
        SOURCE_NAME:    String.
            The name of added plane wave source
        THETA, PHI, POL_ANGLE, AMP, PHASE:  Real scalar.
            Lumerical variable values
        """
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addplane()
        else:
            raise UserWarning('No FDTD solver object could be found.')

        sim.fdtd.set('name', source_name)
        sim.fdtd.setnamed(source_name, 'angle theta', theta)
        sim.fdtd.setnamed(source_name, 'angle phi', phi)
        sim.fdtd.setnamed(source_name, 'polarization angle', pol_angle)
        sim.fdtd.setnamed(source_name, 'amplitude', amp)
        sim.fdtd.setnamed(source_name, 'phase', phase)
        sim.fdtd.setnamed(source_name, 'plane wave type', 'Bloch/periodic') # Use BFAST here for simulation of broad-band plane wave sourc. When considering normal incidence, Bloch would be faster and correct

        sim.fdtd.select(source_name)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = PolarMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.setnamed(source_name, 'direction', direction)

    def make_forward_sim(self, sim):
        """
        This function is used to set the adjoint source before running forawrd simulation
        --INPUT---
        SIM:    Simulation object.
        """

        sim.fdtd.setnamed(self.adjoint_src_names[0], 'enabled', False)
        sim.fdtd.setnamed(self.adjoint_src_names[1], 'enabled', False)

    def make_adjoint_sim(self, sim):
        """
        This function is used to set the adjoint source before running adjoint simulation
        --INPUT---
        SIM:    Simulation object.
        """

        sim.fdtd.setnamed(self.adjoint_src_names[0], 'enabled', True)
        sim.fdtd.setnamed(self.adjoint_src_names[1], 'enabled', True)

    def get_fom(self, sim, nointerpolate=False):
        """
        This function is used to get fom
        ---INPUTS---
        NOINTERPOLATE:
            Boolean. It indicates whether the data from monitor is interpolated (specified position) or not.
        """
        self.wavelength = PolarMatch.get_wavelength(sim)
        trans_coeff, sp_component = PolarMatch.get_transmission_coefficient(sim, self.monitor_name, self.theta, self.phi, self.pol_state, self.wavelength, nointerpolate)
        source_power = PolarMatch.get_source_power(sim, self.wavelength)
        self.T_fwd_vs_wl = np.real(trans_coeff * trans_coeff.conj() / source_power)
        self.phase_prefactors = trans_coeff / 4.0 / source_power
        fom = PolarMatch.fom_wavelength_integral(self.T_fwd_vs_wl, self.wavelength, self.targe_T_forward, self.norm_p)
        return fom, sp_component

    @staticmethod
    def get_wavelength(sim):
        return ModeMatch.get_wavelength(sim)

    @staticmethod
    def get_transmission_coefficient(sim, monitor_name, theta, phi, pol_state, wavelength, nointerpolate, index=1.31):
        """
        This function is used to get the transmission coefficient of the desired polarization
        from the given monitor. Be aware that here we only consider calculating backward transmission coefficient here, and don't assume there is forward propagating components.
        ---INPPUTS---
        INDEX:  Real number.
            Refractive index of the background material, will be used when calculating 'N_m' (plane wave power)
        """
        sp_component = extract_polar(sim.fdtd, monitor_name, wavelength, theta, phi, nointerpolate)
        pol_coeff = np.dot(np.conjugate(pol_state), sp_component)
        N_m = 1/2 * sim.fdtd.getnamed(monitor_name, 'x span') * sim.fdtd.getnamed(monitor_name, 'y span') * index * scipy.constants.speed_of_light * scipy.constants.epsilon_0 * cos(theta)
        # The power of desired plane wave, we assume plane wave to simplify calculation, and assume |E| = 1 , which is true under cwnorm setting
        trans_coeff = pol_coeff * np.sqrt(N_m)
        return trans_coeff.flatten(), sp_component

    @staticmethod
    def get_source_power(sim, wavelength):
        return ModeMatch.get_source_power(sim, wavelength)

    @staticmethod
    def fom_wavelength_integral(T_fwd_vs_wl, wavelength, target_T_forward, norm_p):
        return ModeMatch.fom_wavelength_integral(T_fwd_vs_wl, wavelength, target_T_forward, norm_p)

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        return super().fom_gradient_wavelength_integral(T_fwd_partial_derivs_vs_wl, wl)