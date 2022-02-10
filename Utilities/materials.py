import numpy as np
import scipy as sp
import scipy.constants

from Utilities.wavelength import Wavelength

class Material():
    """
    This class is used to define the permittivity of a material assigned to a geometric primitive. In FDTD, this can be done  in following rwo ways:
        (1) Give the name of material that is accessible from the material database, eg: 'Si(silicon) - Palik'
        (2) Provide the desired epsilon the the geometry primitive directly
    To implement the first way, assign the 'name' property with the desired material name, and base_eps will be ignored automatically.
    To implement the second way, set the 'name' property to the default value ('<Object defined dielectric>'), and assign the desired epsilon value to the property 'base_eps'

    ---INPUTS---
    BASE_EPS: The desires epsilon value of material, while using the second way to implement
    NAME: The name of desired material, when using the first way to implement, and remains the default  
        value when using the second way
    MESH_ORDER: Order of material resolution for overlapping primitives
    """
    object_dielectric = '<Object defined dielectric>'

    def __init__(self, base_eps = 1.0, name = object_dielectric, mesh_order = None) -> None:
        self.base_eps = base_eps
        self.name = name
        self.mesh_order = mesh_order

    def set_script(self, sim, poly_name):
        """This function is used to set the polygon object ('name')'s material property to the given one"""
        sim.fdtd.setnamed('poly_name', 'material', self.name)
        wavelengths = Material.get_wavelengths(sim)
        freq_array = scipy.constants.speed_of_light / wavelengths.asarray()

        if self.name == self.object_dielectric:
            refractive_index = np.sqrt(self.base_eps)
            sim.fdtd.setnamed(poly_name, 'index', refractive_index)
            self.permittivity = self.base_eps * np.ones(freq_array.shape)
        else:
            refractive_index = sim.fdtd.getfdtdindex(self.name, freq_array, freq_array.min(), freq_array.max())
            self.permittivity = np.asarray(np.power(refractive_index, 2)).flatten()
        if self.mesh_order:
            sim.fdtd.setnamed(poly_name, 'override mesh order from material database', True)
            sim.fdtd.setnamed(poly_name, 'mesh order', self.mesh_order)

    @staticmethod
    def get_wavelengths(sim):
        """Create the wavelength object according to the settings of sim"""
        return Wavelength(sim.fdtd.getglobalsource('wavelength start'),
                          sim.fdtd.getglobalsource('wavelength stop'),
                          sim.fdtd.getglobalmonitor('frequency points'))