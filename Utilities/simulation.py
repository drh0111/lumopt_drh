from re import S
import lumapi

class Simulation():
    """
    This class act like the proxy class for FDTD simulation, it is the object to manage the FDTD CAD
    """

    def __init__(self, working_dir, use_var_fdtd, hide_fdtd_cad) -> None:
        """
        Initialize lumerical file and store the handle and working directory.

        ---INPUTS---
        WORKING_DIR: String object. It represent the working directory that the lumerical file 
            will be stored in.
        USE_VAR_FDTD: Boolean object. Whether use varFDTD or not (FDTD).
        HIDE_FDTD_CAD: Boolean object. Whether hide the FDTD CAD and run the simulation in 
            background or not.
        """

        self.fdtd = lumapi.MODE(hide = hide_fdtd_cad) if use_var_fdtd else lumapi.FDTD(hide = hide_fdtd_cad)
        self.working_dir = working_dir
        self.fdtd.cd(working_dir)

    def run(self, name, num):
        """
        Save the simulation file in the desired working directory and run the simulation
        """
        self.fdtd.cd(self.working_dir)
        self.fdtd.save("{}_{}".format(name, num))
        self.fdtd.run()

    def save(self):
        """remove the result (by switchtolayout) and save the result"""
        self.fdtd.switchtolayout()
        self.fdtd.cd(self.working_dir)
        self.fdtd.save()

    def __del__(self):
        self.fdtd.close()