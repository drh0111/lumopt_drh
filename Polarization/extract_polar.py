import numpy as np
import scipy.constants as C
import lumapi
from Lumerical_methods.lumerical_scripts import get_field

def extrac_polar(fdtd, monitor_name, omega, k_p):
    """
    This function is used to calculate the transmission coefficient of the right circularly polarized light at the K_P direction from the given monitor's field distribution, the program has considered the case that the field has multifrequency

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
        Normalized two-element real-valued array. It represents the k parallel direction which we are interested in and will calculate the transmission coefficient, in SI unit
    XX: 
        2D array. meshgrid of x
    YY:
        2D array. meshgrid of y

    ---OUTPUT---
    F_AVE:
        2D array, 2 dimensions are respectively 'frequency' and 'vector component'
    """

    kernel = np.exp(-1j *(k_p[0] * xx + k_p[1] * yy))
    F_ave = np.mean(kernel * np.transpose(fields), (2, 3)) # transpose to used broadcast
    return np.transpose(F_ave)

def s_p_coeff(F_ave, omega, k_p):
    """
    This function is used to calculate the s and p component of the average field. The definition of the s and p polarized light follows the standard defined in the SI of paper 'Singular points of polarizations in the momentum space of photonic crystal slabs'
    ---INPUTS---
    F_AVE:
        2D array, 2 dimensions are respectively 'frequency' and 'vector component'
    OMEGA:
        Number, angular frequnecy in dimension Hz*rad
    K_P:
        Normalized two-element real-valued array. It represents the k parallel direction which we are interested in and will calculate the transmission coefficient, in SI unit
    ---OUTPUT---
    OUTPUT:
        2D array, with size (2, num) storing the s and p components of light respectively, corresponding to the x and y component in source axis (see Lumerical reference)
    """
    if np.linalg.norm(k_p) < 0:
        norm_k =  omega / C.speed_of_light
        kz = -np.sqrt(norm_k**2 - k_p[0]**2 - k_p[1]**2)
        k = np.array([k_p[0], k_p[1], kz])

        z_norm = np.array([0, 0, 1])
        k_norm = k / np.linalg.norm(k)
        s_vector = np.cross(z_norm, k_norm)
        s_vector = s_vector / np.linalg.norm(s_vector)
        p_vector = np.cross(k_norm, s_vector)
        p_vector = p_vector / np.linalg.norm(p_vector)

        s_component = np.dot(F_ave, s_vector)
        p_component = np.dot(F_ave, p_vector)
    else:
        # when k_p's norm is 0, the S polarization reduce to Y polarization, and P poalrization reduce to X polarization, according to the coordinate of the Lumerical FDTD
        s_component = F_ave[:, 1]
        p_component = F_ave[:, 0]

    return np.array([s_component, p_component])