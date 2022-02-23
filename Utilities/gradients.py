import numpy as np
import scipy as sp
import scipy.constants
import lumapi


class GradientFields():
    """
    This class is used to calculate the gradients of fom with respect to the parameters, given the forward fields and adjoint fields
    """
    def __init__(self, forward_fields, adjoint_fields) -> None:
        """
        check the correctness of input and initialize the class
        ---INPUTS---
        FORWARD_FIELDS: Fields or FieldsNoInterp class, it contains the fields of forward 
            simulation.
        ADJOINT_FIELDS: Fields or FieldsNoInterp class, it contains the fields of adjoint 
            simulation.
        """
        error_str = 'forward fields parameters and adjoint fields parameters don not fit'
        assert forward_fields.x.shape == adjoint_fields.x.shape, error_str
        assert forward_fields.y.shape == adjoint_fields.y.shape, error_str
        assert forward_fields.z.shape == adjoint_fields.z.shape, error_str
        assert forward_fields.wl.shape == adjoint_fields.wl.shape, error_str

        self.forward_fields = forward_fields
        self.adjoint_fields = adjoint_fields

    def boundary_perturbation_integrand(self):
        """
        This function is used to generate the integral kernel of the equation 5.28 in Owen Miller's thesis used to calculate the partial derivatives of the FOM versus the parameters to be optimized
        """
        def project(a, b):
            """This function will project vector a on vector b"""
            norm_b = b / np.linalg.norm(b)
            return np.dot(a, norm_b) * norm_b

        def gradient_field(x, y, z, wl, normal, eps_in, eps_out):
            """
            This function will calculate and return the kernel integral for the given arguments
            ---INPUTS---
            X:  Scalar, x position
            Y:  Scalar, y position
            Z:  Scalar, z position
            WL: Scalar, expected wavelength
            NORAML: 1-D array, the normal direction pointing the outside
            EPS_IN: Scalar, the epsilon for inside material
            EPS_OUT:    Scalar, the epsilon for outside material
            """
            E_forward = self.forward_fields.getfield(x, y, z, wl)
            D_forward = self.forward_fields.getDfield(x, y, z, wl)
            E_adjoint = self.adjoint_fields.getfield(x, y, z, wl)
            D_adjoint = self.adjoint_fields.getDfield(x, y, z, wl)
            E_forward_parallel = E_forward - project(E_forward, normal)
            E_adjoint_parallel = E_adjoint - project(E_adjoint, normal)
            D_forward_perp = project(D_forward, normal)
            D_adjoint_perp = project(D_adjoint, normal)
            result = 2 * (scipy.constants.epsilon_0 * (eps_in - eps_out) * np.sum(E_forward_parallel * E_adjoint_parallel) + (1/eps_out - 1/eps_in) / scipy.constants.epsilon_0 * np.sum(D_forward_perp * D_adjoint_perp))
            return np.real(result)
        return gradient_field

    @staticmethod
    def spatial_gradient_integral_on_cad(sim, forward_fields, adjoint_fields, wl_scaling_factor):
        """
        This function is used to calculate the fom gradients vs wavelength in the dir_grad == True case (which is done by integrating over the space), it should be used after calculating the fowrward fields, backward fields and d_epses
        ---INPUTS---
        FORWARD_FIELDS: String.
            Name of variable in Lumerical that store data of the forward fields
        ADJOINT_FIELDS: String.
            Name of variable in Lumerical that store data of the adjoint fields
        WL_SCALING_FACTOR:  1-D array.
            It stores the scaling factor of adjoint fields vs wavelength
        """
        lumapi.putMatrix(sim.fdtd.handle, 'wl_scaling_factor', wl_scaling_factor)
        sim.fdtd.eval("grad_fields = 2.0 * eps0 * {0}.E.E * {1}.E.E;".format(forward_fields, adjoint_fields))
        sim.fdtd.eval("num_wl = length(wl_scaling_factor); num_params = length(d_epses);")
        sim.fdtd.eval("partial_fom_derivs_vs_lambda = matrix(num_wl, num_params);" +
                    "   for (i_wl = [1 : num_wl]){" + 
                    "       for(i_param = [1 : num_params]){" +
                    "           spatial_integrand = pinch(sum(grad_fields(:, :, :, i_wl, :) * d_epses{i_param} * wl_scaling_factor, 5), 4);" +
                    "           partial_fom_derivs_vs_lambda(i_wl, i_param) = integrate2(spatial_integrand, [1 : 3], {0}.E.x, {0}.E.y, {0}.E.z);".format(forward_fields) + 
                    "       }" + 
                    "   }")
        partial_fom_derivs_vs_lambda = lumapi.getVar(sim.fdtd.handle, "partial_fom_derivs_vs_lambda")
        sim.fdtd.eval("clear(wl_scaling_factor, grad_fields, partial_fom_derivs_vs_lambda, num_wl, num_params, spatial_integrand, d_epses, {0}, {1});".format(forward_fields, adjoint_fields))
        return partial_fom_derivs_vs_lambda