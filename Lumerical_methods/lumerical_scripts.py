import numpy as np
import scipy as sp
import scipy.constants

import lumapi
from Utilities.fields import Fields, FieldsNoInterp

def get_field(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, noninterpolation, unfold_symmetry = True, on_cad_only = False):
    """
    This function is used to get the data of field (E, D, H, eps) from the Limerical FDTD, and store it in python variables.
    ---INPUTS---
    FDTD:   FDTD object.
    MONITOR_NAME:   String, the name of monitor we will get fields from
    FIELD_RESULT_NAME:  String, The name of the struct variable in Lumerical that store the raw 
        fields
    GET_EPS:    Bool, Whether get epsilon or not
    GET_D:  Bool, Whether get D field or not
    GET_H:  Bool, Whether get H field or not
    NONINTERPOLATION:   Bool, whether
    UNFOLD_SYMMETRY:    Bool,
    ON_CAD_ONLY:    Bool, Whether store the desired field data in CAD only or also in python
    ---OUTPUT---
    RETURN: Field class, it contains the interpolated fields
    """
    get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, noninterpolation, unfold_symmetry)

    if not on_cad_only:
        fields_dict = lumapi.getVar(fdtd.handle, field_result_name)
        
        # make some modification to eps according to the setup of the simulation
        if get_eps:
            if fdtd.getnamednumber('varFDTD') == 1:
                if 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and not ('index_z' in fields_dict['index']): # the TE mode in varFDTD simulation
                    fields_dict['index']['index_z'] = fields_dict['index']['index_x'] * 0.0 + 1.0
                elif not ('index_x' in fields_dict['index']) and not ('index_y' in fields_dict['index']) and 'index_z' in fields_dict['index']: # the TM mode in varFDTD
                    fields_dict['index']['index_x'] = fields_dict['index']['index_z'] * 0.0 + 1.0
                    fields_dict['index']['index_y'] = fields_dict['index']['index_x']

            assert 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']

            fields_eps = np.stack([np.power(fields_dict['index']['index_x'], 2),
                                    np.power(fields_dict['index']['index_y'], 2),
                                    np.power(fields_dict['index']['index_z'], 2)],
                                    axis = -1)
        else:
            fields_eps = None

        fields_D = fields_dict['E']['E'] * fields_eps * scipy.constants.epsilon_0 if get_D else None
        fields_H = fields_dict['H']['H'] if get_H else None

        if noninterpolation:
            deltas = [fields_dict['delta']['x'], fields_dict['delta']['y'], fields_dict['delta']['z']]
            return FieldsNoInterp(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], deltas, fields_dict['E']['E'], fields_D, fields_H, fields_eps)
        else:
            return Fields(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], fields_dict['E']['E'], fields_D, fields_H, fields_eps)


def get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, noninterpolation, unfold_symmetry = True):
    """
    This function is used to get the data of field (E, D, H, eps) from the Limerical FDTD, but yet still store in variables in Lumerical FDTD
    ---INPUTS---
    FDTD:   FDTD object.
    MONITOR_NAME:   String, the name of monitor we will get fields from
    FIELD_RESULT_NAME:  String, The name of the struct variable that store the raw fields
    GET_EPS:    Bool, Whether get epsilon or not
    GET_D:  Bool, Whether get D field or not
    GET_H:  Bool, Whether get H field or not
    NONINTERPOLATION:   Bool, whether
    UNFOLD_SYMMETRY:    Bool,
    """
    unfold_symmetry_str = 'true' if unfold_symmetry else 'false' # be attention: no capital letter
    fdtd.eval('options = struct; options.unfold = {};'.format(unfold_symmetry_str) + 
                '{} = struct;'.format(field_result_name) + '{0}.E = getresult("{1}", "E", options);'.format(field_result_name, monitor_name))

    if get_eps or get_D:
        index_monitor_name = monitor_name + '_index'
        fdtd.eval('{0}.index = getresult("{1}", "index", options);'.format(field_result_name, index_monitor_name))

    if get_H:
        fdtd.eval('{0}.H = getresult("{1}", "H", options);'.format(field_result_name, monitor_name))
    
    if noninterpolation:
        fdtd.eval('{}.delta = struct;'.format(field_result_name) + 
                    '{0}.delta.x = getresult("{1}", "delta_x", options);'.format(field_result_name, monitor_name) + 
                    '{0}.delta.y = getresult("{1}", "delta_y", options);'.format(field_result_name, monitor_name))

        if fdtd.getresult(monitor_name, 'dimension') == 3:
            fdtd.eval('{0}.delta.z = getresult("{1}", "delta_z", options);'.format(field_result_name, monitor_name))
        else:
            fdtd.eval('{}.delta.z = 0.0;'.format(field_result_name))

def set_spatial_interp(fdtd, monitor_name, method):
    """
    This function is used to set the desired monitor's interpolation method
    """
    script = 'select("{}"); set("spatial interpolation", "{}")'.format(monitor_name, method)
    fdtd.eval(script)