import numpy as np

class Edge():
    """
    This class is used to describe an side-edge of the 3D geometry that is extruded form a 2D geometry

    ---INPUTS---
    FIRST_POINT: 2 dimensional column array, representing the position of the first point of the edge
    """

    def __init__(self, first_point, second_point, eps_in, eps_out, z, depth) -> None:
        self.first_point = first_point
        self.second_point = second_point
        self.eps_in = eps_in
        self.eps_out = eps_out
        self.z = float(z)
        self.depth = float(depth)

        normal_vec = np.flipud(first_point - second_point)
        normal_vec = np.array([-normal_vec[0], normal_vec[1], 0])
        self.norm = normal_vec / np.linalg.norm(normal_vec)
    pass