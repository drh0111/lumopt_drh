import copy
import numpy as np

class Optimizer:
    """ The base class of all optimizers, define some shared functions"""
    def __init__(self, max_iter, scaling_factor, target_fom) -> None:
        """
        ---INPUT---
        MAX_ITER:   Positive scalar, maximum number of iterations in the optimization.
        SCALING_FACTOR:    Scalar or row vector that fits the length of the optimized parameters, it is 
            used to scale the magnitude of parameters to 1, which is beneficial for the optimization as most optimizers assume the variables to optimize are roughly of order of 1
        TARGET_FOM: A scalar, the value of the target figure of merit.
        """
        self.max_iter = max_iter
        self.scaling_factor = np.array(scaling_factor).flatten()
        self.target_fom = target_fom
        self.current_fom = None
        self.current_params = []
        self.current_gradients = []
        self.fom_hist = []
        self.params_hist = []
        self.gradients_hist =[]
        self.iter = 0
        self.fom_scaling_factor = 1e12

    def initialize(self, start_params, callable_fom, callable_jac, bounds, plotting_fun):
        """
        __init__ function is used to make an original initialization, this function is used to make a more comprehensive initialization that use some variables form other objects collected by the Optimization object. Be aware that it will scale the parameters and bounds here.

        ---INPUTS---
        START_PARAMS: Row vector (numpy array), without scaling
        BOUNDS: Parameters are aligned in column direction
        PLOTTING_FUN:   Plotter class.
            It is responsilble for plotting the data field during the optimization
        """
        assert bounds.shape[0] == start_params.size and bounds.shape[1] == 2
        assert self.scaling_factor.size == 1 or self.scaling_factor.size == start_params.size
        self.callable_fom, self.callable_jac = self.define_callables(callable_fom, callable_jac)
        self.bounds = bounds * self.scaling_factor.reshape((self.scaling_factor.size, 1))
        self.callback = self.define_callback(plotting_fun)
        self.reset_start_point(start_params)

    def reset_start_point(self, start_params):
        """
        This function is used to set the initial point (after scaling) of optimization
        """
        assert self.bounds.shape[0] == start_params.size 
        self.start_point = np.array(start_params) * self.scaling_factor

    def define_callables(self, callable_fom, callable_jac) -> callable:
        """
        This function is used to define callable functions to return fom and gradients in the optimization, considering the scaling of variables and fom. Be aware that fom_scaling_factor is only for the convenience of plotting, fom of object is unchanged

        ---INPUT---
        CALLABLE_FOM: Original function that takes numpy vector and returns fom
        CALLABLE_JAC: Original function that takes numpy vector and returns gradients  (row vector)

        ---OUTPUT---
        Two adjusted function handlers
        """
        def callable_fom_local(params):
            # params shold be numpy array
            fom = callable_fom(params / self.scaling_factor)
            self.current_fom = fom
            return self.current_fom * self.fom_scaling_factor

        def callable_jac_local(params):
            # params shold be numpy array
            gradients = callable_jac(params / self.scaling_factor) / self.scaling_factor # A problem: whether need / scaling_factor here?
            self.current_gradients = gradients
            return self.current_gradients * self.fom_scaling_factor

        return callable_fom_local, callable_jac_local

    def define_callback(self, plot_func) -> callable:
        """
        This function is used to define callback function that will be called at the end of each iteration, for the purpose of record, update and show

        ---INPUT---
        PLOT_FUNC: function that can plot the designed structure out
        """
        def callback(*arg):
            self.params_hist.append(copy.copy(self.current_params))
            self.fom_hist.append(self.current_fom)
            self.gradients_hist.append(copy.copy(self.current_gradients))
            self.iter += 1
            plot_func()
            self.report_writing()
        return callback
    
    def report_writing(self):
        """
        This function is to recorf the history of optimization
        """
        with open('optimization_report.txt','a') as f:
            f.write('Iteration {}: FOM = {}\n'.format(self.iter, self.current_fom))
            f.write('Parameters: {}\n'.format(self.current_params / self.scaling_factor))
            f.write('\n\n')

    def plot(self, fom_ax, params_ax = None, gradients_ax = None):
        """
        This function is used to plot the history of fom, params and gradient during the optimization 
        """
        fom_ax.clear()
        fom_hist = np.abs(self.fom_hist)
        if self.target_fom == 0:
            fom_ax.plot(range(self.iter), fom_hist) # take absolute value for visualization
        else:
            fom_ax.semilogy(range(self.iter), fom_hist)
        fom_ax.set_xlabel('Iteration')
        fom_ax.set_title('Figure of Merit')
        fom_ax.set_ylabel('FOM')

        if params_ax is not None:
            # params_hist = np.transpose(np.abs(self.params_hist)) # different from the standard one, transpose here for the implementation of several curve by one command
            params_ax.clear()
            params_ax.semilogy(range(self.iter), np.abs(self.params_hist))
            params_ax.set_xlabel('Iteration')
            params_ax.set_ylabel('Parameters')
            params_ax.set_title('Parameter evolution')

        if gradients_ax is not None:
            # grad_hist = np.transpose(np.abs(self.gradients_hist))
            gradients_ax.clear()
            gradients_ax.semilogy(range(self.iter), np.abs(self.gradients_hist))
            gradients_ax.set_xlabel('Iteration')
            gradients_ax.set_ylabel('Gradient Magnitude')
            gradients_ax.set_title('Gradient evolution')
