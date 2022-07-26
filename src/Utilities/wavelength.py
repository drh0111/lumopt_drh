import numpy as np

class Wavelength():
    """
    This class is used to set the wavelength of concerned during the optimization
    """

    def __init__(self, start, stop = None, points = 1):
        self.start = float(start)
        self.stop = float(start) if stop is None else float(stop)
        if self.stop < self.start:
            raise UserWarning('stop should be bigger that start')
        self.points = int(points)
        if self.points < 1:
            raise UserWarning('points should be positive integer')
        if self.start == self.stop and self.points > 1:
            raise UserWarning('multiple points with zero wavelength span')

    def min(self):
        """Return minimum wavelength"""
        return self.start

    def max(self):
        """Returning maximum wavelength"""
        return self.stop

    def __len__(self):
        """Enable the len() method, and return number of points"""
        return self.points

    def __getitem__(self, item):
        """Enable the getitem method, return the wavelength of given order"""
        return self.asarray()[item]

    def asarray(self):
        """Return an array of wavelength that is under concern and simulation"""
        return np.linspace(self.start, self.stop, self.points)

