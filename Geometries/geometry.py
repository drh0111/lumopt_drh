import sys
import numpy as np
import lumapi

class Geometry():
    unfold_symmetry = True # We will unfold the symmetry of the monitor by default

    def use_interpolation(self):
        """Given the flag showing whether use interpolation or not"""
        return False