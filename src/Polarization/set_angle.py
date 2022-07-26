import numpy as np

def set_angle(fdtd, src_name, theta, phi):
    """
    This function is used to set the angles of sources in the simulation and will be used for loop

    ---INPUTS---
    FDTD:
        Simulation handle
    SOURCE_NAME:
        String, the name of source that will be modified
    THETA:
        Number, the theta angle of the x and y polarized source in FDTD in degree
    PHI:
        Number, the phi angle of the x and y polarized source in FDTD in degree
    ---OUTPUT---
    2D array
    KX:
        Number, the KX of the simulation in bandstructure unit
    KY:
        Number, the KY of the simulation in bandstructure unit
    """

    fdtd.switchtolayout()
    fdtd.setnamed(src_name, 'angle theta', theta)
    fdtd.setnamed(src_name, 'angle phi', phi)
    kx = fdtd.getnamed('FDTD', 'kx')
    ky = fdtd.getnamed('FDTD', 'ky')

    return np.array([kx, ky])