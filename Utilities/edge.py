import numpy as np

from Utilities.gradients import GradientFields

class Edge():
    """
    This class is used to describe an side-edge of the 3D geometry that is extruded form a 2D geometry

    ---INPUTS---
    FIRST_POINT:    1 dimensional array, representing the position of the first point of the edge
    SECOND_POINT:   1 dimensional array, representing the position of the first point of the edge
    EPS_IN: Scalar or Material class, epsilon of the polygon material
    EPS_OUT:    Scalar or Material class, epsilon of the material outside the polygon
    """

    def __init__(self, first_point, second_point, eps_in, eps_out, z, depth) -> None:
        self.first_point = first_point
        self.second_point = second_point
        self.eps_in = eps_in  # in and out might be indicated by normal_vec
        self.eps_out = eps_out
        self.z = float(z)
        self.depth = float(depth)

        normal_vec = np.flipud(first_point - second_point)
        normal_vec = np.array([-normal_vec[0], normal_vec[1], 0])
        self.normal = normal_vec / np.linalg.norm(normal_vec)

    def derivative(self, gradient_fields, n_points):
        """
        This function is used to calculate the derivative for moving the two extremity points of an edge in the direction normal to the edge for the 2D and 3D cases
        """
        if len(gradient_fields.forward_fields.z) == 1:
            return self.derivative_2D(gradient_fields, n_points)
        else:
            return self.derivative_3D(gradient_fields, n_points)

    def derivative_3D(self, gradient_fields, n_points):
        """
        This function is used to calculate the derivative for moving the two extremity points of an edge in the direction normal to the edge for the 3D case by considering the depth
        ---INPUTS---
        GRADIENT_FIELDS:    The GradientFields class.
            Contain the data of forward_fields and adjoint_fields
        N_POINTS:   Integer number.
            it is the number of quadrature points in the integral
        """
        edge_derivs_2D = self.derivative_2D(gradient_fields, n_points)
        deriv_first = edge_derivs_2D[0] * self.depth
        deriv_second = edge_derivs_2D[1] * self.depth
        return (deriv_first, deriv_second)

    def derivative_2D(self, gradient_fields, n_points):
        """
        This function is used to calculate the derivative for moving the two extremity points of an edge in the direction normal to the edge for the 2D case
        ---INPUTS---
        GRADIENT_FIELDS:    The GradientFields class.
            Contain the data of forward_fields and adjoint_fields
        N_POINTS:   Integer number.
            it is the number of quadrature points in the integral
        ---OUTPUTS---
        """
        points_along_edge_on_unity_scale = np.linspace(0, 1, n_points)
        points_along_edge_interp_fun = lambda r: (1 - r) * self.first_point + r * self.second_point
        points_along_edge = list(map(points_along_edge_interp_fun, points_along_edge_on_unity_scale))
        # integrand in (5.28) of Owen Miller's thesis
        integrand_interp_fun = gradient_fields.boundary_perturbation_integrand()
        wavelengths = gradient_fields.forward_fields.wl
        eps_in = self.eps_in.get_eps(wavelengths)
        eps_out = self.eps_out.get_eps(wavelengths)
        integrand_along_edge = list()
        for idx, wl in enumerate(wavelengths):
            integrand_along_edge_fun = lambda point: integrand_interp_fun(point[0], point[1], self.z, wl, self.normal, eps_in[idx], eps_out[idx])
            integrand_along_edge.append(list(map(integrand_along_edge_fun, points_along_edge)))
        integrand_along_edge = np.array(integrand_along_edge).transpose().squeeze()
        tangent_vec_length = np.sqrt(np.sum(np.power(self.first_point - self.second_point, 2)))
        # integrate to get derivative at the second edge (in normal direction)
        weights = np.outer(points_along_edge_on_unity_scale, np.ones(len(wavelengths))).squeeze()
        deriv_second = np.trapz(y = weights * integrand_along_edge, x = tangent_vec_length * points_along_edge_on_unity_scale, axis = 0)
        # integrate to get derivative at the second edge (in normal direction)
        flipped_weights = np.flip(weights, axis = 0)
        deriv_first = np.trapz(y = flipped_weights * integrand_along_edge, x = tangent_vec_length * points_along_edge_on_unity_scale, axis = 0)
        # derivatives at both endpoints
        return [deriv_first, deriv_second]