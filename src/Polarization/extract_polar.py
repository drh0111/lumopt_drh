import numpy as np
import scipy.constants as C
from cmath import cos, sin
import lumapi
from Lumerical_methods.lumerical_scripts import get_field

def extract_polar(fdtd, monitor_name, wavelengths, theta, phi, nointerpolate):
    """
    This function is used to calculate the transmission coefficient of s, p polarized light at the given input direction (THETA, PHI) from the given monitor's field distribution, the program has considered the case that the field has multifrequency

    ---INPUTS---
    FDTD:   
        Simulation handle
    MONITOR_NAME:   
        String. Monitor's name from which we will calculate the transmission coefficient
    WAVELENGTHS:
        1D array. The wavelength points this function will consider (simulation points), in 'm' unit.
    THETA, PHI:
        Both real number. They define the incident angle of source. We follows the standard in Lumerical FDTD, and only consider Backward direction in z axis here.
    NOINTERPOLATE:
        Boolean. It indicates whether the data from monitor is interpolated (specified position) or not.
    ---OUTPUT---
     OUTPUT:
        2D array, with size (2, num) storing the s-pol and p-pol components of light respectively
    """
    # Check the simulation's interpolation
    interpolation = fdtd.getnamed(monitor_name, 'spatial interpolation')
    if (nointerpolate == True and interpolation != 'none') or (nointerpolate == False and interpolation != 'specified position'):
        raise UserWarning('Nointerpolation setting is inconsistent with simulation')
    # Be aware that use nointerpolation in the setting of the simulation
    fields = get_field(fdtd, monitor_name, field_result_name = 'fields', get_eps = False, get_D = False, get_H = False, noninterpolation = nointerpolate)
    E_fields = fields.E[:, :, 0, :, :]
    k_p = np.array([-2*np.pi/wavelengths * sin(theta)*cos(phi), -2*np.pi/wavelengths * sin(theta)*sin(phi)])


    if nointerpolate == True:
        pass
    else:
        xx, yy = np.meshgrid(fields.x, fields.y)
        E_ave = average_field(E_fields, k_p, xx, yy)
        sp_component = s_p_coeff(E_ave, wavelengths, k_p)

    return sp_component


def extrac_polar_lr(fdtd, monitor_name, omega, k_p):
    """
    This function is used to calculate the transmission coefficient of the left and right circularly polarized light at the K_P direction from the given monitor's field distribution, the program has considered the case that the field has multifrequency

    ---INPUTS---
    FDTD:   
        Simulation handle
    MONITOR_NAME:   
        String. Monitor's name from which we will calculate the transmission coefficient
    K_P:
        Normalized two-element real-valued array. It represents the k parallel direction which we are interested in and will calculate the transmission coefficient, in SI unit
    OMEGA:
        Number, angular frequnecy in dimension Hz*rad
    ---OUTPUT---
     OUTPUT:
        2D array, with size (2, num) storing the LCP and RCP components of light respectively
    """
    # Jones representation of the RCP and LCP, with the first component as x and the second as y with respect to the source axis (see Lumerical reference)
    rcp = 1/np.sqrt(2) * np.array([1, 1j]) 
    lcp = 1/np.sqrt(2) * np.array([1, -1j])
    # Problem: The standard of i and j here

    # Be aware that use nointerpolation in the setting of the simulation
    fields = get_field(fdtd, monitor_name, field_result_name = 'fields', get_eps = False, get_D = False, get_H = False, noninterpolation = False)
    E_fields = fields.E[:, :, 0, :, :]

    # form integral kernel to select the desired K_P
    xx, yy = np.meshgrid(fields.x, fields.y)
    E_ave = average_field(E_fields, k_p, xx, yy)
    sp_component = s_p_coeff(E_ave, omega, k_p)
    l_component = np.dot(rcp, sp_component) # be aware to use rcp here because of conjugate
    r_component = np.dot(lcp, sp_component)
    return np.array([l_component, r_component])
    

def average_field(fields, k_p, xx, yy):
    """
    This function is used to get the average fields of the expected K_P, i.e. the polarization at that direction
    ---INPUTS---
    FIELDS:
        4D array. 4 dimensions are respectively 'x', 'y', 'frequency', 'vector component'
    K_P:
        Normalized 2-D real-valued array with size [2, num_freq]. It represents the k parallel direction which we are interested in and will calculate the transmission coefficient, in SI unit
    XX: 
        2D array. meshgrid of x
    YY:
        2D array. meshgrid of y

    ---OUTPUT---
    F_AVE:
        2D array, 2 dimensions are respectively 'frequency' and 'vector component'
    """
    num_freq = k_p.shape[1]
    xx = np.transpose(np.expand_dims(xx, 0).repeat(num_freq, 0)) # transpose to used broadcast
    yy = np.transpose(np.expand_dims(yy, 0).repeat(num_freq, 0))
    kernel = np.exp(-1j *(k_p[0, :] * xx + k_p[1, :] * yy))
    F_ave = np.mean(np.expand_dims(kernel, 3) * fields, (0, 1))
    return F_ave

def s_p_coeff(F_ave, wavelengths, k_p):
    """
    This function is used to calculate the s and p component of the average field. The definition of the s and p polarized light follows the standard defined in the SI of paper 'Singular points of polarizations in the momentum space of photonic crystal slabs'
    ---INPUTS---
    F_AVE:
        2D array, 2 dimensions are respectively 'frequency' and 'vector component'
    WAVELENGTHS:
        Number or 1-D array, representing selected wavelength in m unit
    K_P:
        Normalized 2-D real-valued array with size [2, num_freq]. It represents the k parallel direction which we are interested in and will calculate the transmission coefficient, in SI unit
    ---OUTPUT---
    OUTPUT:
        2D array, with size (2, num) storing the s and p components of light respectively, corresponding to the x and y component in source axis (see Lumerical reference)
    """
    if np.linalg.norm(k_p[:, 0]) > 0: # Only consider all k_p equal 0 or not
        norm_k =  2*np.pi / wavelengths
        kz = -np.sqrt(norm_k**2 - k_p[0]**2 - k_p[1]**2)
        k = np.array([k_p[0], k_p[1], kz])

        z_norm = np.array([0, 0, 1])
        k_norm = k / np.linalg.norm(k, axis=0)
        s_vector = np.cross(z_norm, k_norm, axis=0)
        s_vector = s_vector / np.linalg.norm(s_vector, axis=0)
        p_vector = np.cross(k_norm, s_vector, axis=0)
        p_vector = p_vector / np.linalg.norm(p_vector, axis=0)

        s_component = (F_ave * np.transpose(s_vector)).sum(1)
        p_component = (F_ave * np.transpose(p_vector)).sum(1)
    else:
        # when k_p's norm is 0, the S polarization reduce to Y polarization, and P poalrization reduce to X polarization, according to the coordinate of the Lumerical FDTD
        s_component = F_ave[:, 1]
        p_component = F_ave[:, 0]

    return np.array([s_component, p_component])