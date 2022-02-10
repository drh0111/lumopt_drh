import os

from Utilities.load_lsf import load_lsf
from lumapi import FDTD
from lumapi import MODE

class BaseScript():
    """
    This is the proxy class that stands as the interface to make the calling from users implemented appropriately in the FDTD CAD, in order to make the simulatation run safely and correctly
    """
    def __init__(self, str_obj) -> None:
        """
        ---INPUTS---
        STR_OBJ: It ca be:
            (1) Absolute address of any visible Lumerical project file (*.fsp)
            (2) Absolute address of any visible Lumerical scirpt file (*.lsf)
            (3) Plain string written under Lumerical script commands
        """
        # Check the input
        if not isinstance(str_obj, str):
            raise UserWarning('The input must be string typed object')

        if ('.fsp' in str_obj and os.path.isfile(str_obj)):
            self.project = str_obj
        elif ('.lsf' in str_obj and os.path.isfile(str_obj)):
            self.script = load_lsf(str_obj)
        else:
            self.script = str_obj

    def __call__(self, cad_handle):
        """Check the handle and evaluate the base_script in the handle"""
        return self.eval(self, cad_handle)

    def eval(self, cad_handle):
        """
        This funciton is used to load the desired project file or lumerical script
        ---INPUTS---
        CAD_HANDLE: Handle of FDTD or MODE
        """
        if not isinstance(cad_handle, FDTD) or not isinstance(cad_handle, MODE):
            raise UserWarning('input must be handle of FDTD or MODE')
        
        if hasattr(self, 'project'):
            return cad_handle.load(self.project)
        elif hasattr(self, 'script'):
            return cad_handle.eval(self.script)
        else:
            raise UserWarning('un-initialized object')