import scipy as sp
import scipy.optimize as spo

from Optimizers.optimizer import Optimizer

class ScipyOptimizers(Optimizer):
    """
    This class of optimizer is based on the SciPy's optimization package, for official information, please refer to:
                     https://docs.scipy.org/doc/scipy/reference/optimize.html#module-scipy.optimize
    L-BFGS-G method is used here to do the optimization, which approximate the Hessian from the different optimization steps

    ---INPUTS---
    MAX_ITER:   A scalar, maximum number of iteration
    METHOD: String, the method used in optimization
    SCALING_FACTOR: Scalar or row vector that fits the length of the optimized parameters, it is used to scale the  
        magnitude of parameters to 1, which is beneficial for the optimization as most optimizers assume the variables to optimize are roughly of order of 1
    PGTOL:  A scalar, the tolerance of projected gradient method, see 'L-BFGS-B' document for more 
        information
    FTOL:   A scalar, the tolerance of FOM in the optimization (when the change is smaller than this, 
        optimization will stop)
    TARGET_FOM: A scalar, the value of the target figure of merit
    """

    def __init__(self, max_iter, method = 'L-BFGS-B', scaling_factor = 1.0, pgtol = 1.0e-5, ftol = 1.0e-12, target_fom = 0) -> None:
        super(ScipyOptimizers, self).__init__(max_iter, scaling_factor, target_fom)
        self.method = str(method)
        self.pgtol = float(pgtol)
        self.ftol = float(ftol)

    def define_callables(self, callable_fom, callable_jac) -> callable:
        """
        This function is used to define callable functions that optimizers will use to evaluate fom and gradients. The sign of the fom and gradients are changed to make the overall optimization a maximization problem.  considering the scaling of variables and fom. Be aware that fom_scaling_factor is only for the convenience of plotting, fom of object is unchanged

        ---INPUT---
        CALLABLE_FOM: Original function that takes numpy vector and returns fom
        CALLABLE_JAC: Original function that takes numpy vector and returns gradients (row vector)

        ---OUTPUT---
        Two adjusted function handlers
        """
        def callable_fom_local(params):
            # params shold be numpy array
            fom = callable_fom(params / self.scaling_factor)
            self.current_params = params
            self.current_fom = self.target_fom - fom # change the sign of fom here
            return self.current_fom * self.fom_scaling_factor

        def callable_jac_local(params):
            # params should be numpy array
            gradients = callable_jac(params / self.scaling_factor) / self.scaling_factor
            self.current_gradients = -gradients
            return -gradients * self.fom_scaling_factor

        return callable_fom_local, callable_jac_local

    def run(self):
        """
        This is the function run by Optimization object to do the whole optimization process using SciPy's optimization package
        """
        print('Run the scipy optimization \n')
        print('Bounds = {} \n'.format(self.bounds))
        print('Start = {} \n'.format(self.start_point))
        res = spo.minimize(fun = self.callable_fom,
                            x0 = self.start_point,
                            jac = self.callable_jac,
                            bounds = self.bounds,
                            callback = self.callback,
                            options = {'maxiter':self.max_iter, 'disp':True, 'gtol':self.pgtol, 'ftol':self.ftol},
                            method = self.method)
        res.x /= self.scaling_factor
        res.fun, res.jac = -res.fun, -res.jac * self.scaling_factor
        return res