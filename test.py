"""
This is a test function to test the correctness of optimizers
"""
import numpy as np
import matplotlib.pyplot as plt

from Optimizers.generic_optimizers import ScipyOptimizers

def f(x):
    # x is a two element column vector
    return -(np.exp(np.cos(x[0]) + np.cos(x[1])))
def g(x):
    # return the gradients (row vector) of f
    gradients = [-np.sin(x[0]) * f(x), 
                 -np.sin(x[1]) * f(x)]
    return np.array(gradients)

def plot(position = None):
    # 'position' should be array of dimension (num, 2)
    if position is None:
        pass
    else:
        # Construct x, y axis mesh grid for contour map
        num = 1000
        x = np.linspace(2, 4, num)
        y = np.linspace(1.5, 4, num)
        X, Y = np.meshgrid(x, y)
        var = np.concatenate((X.reshape((1, -1)), Y.reshape(1, -1)))
        Z = -f(var)
        Z = Z.reshape(X.shape)

        # Draw the contour picture
        plt.figure(figsize = (10, 6))
        plt.contourf(X, Y, Z)
        plt.contour(X, Y, Z)

        # Draw the points
        px = position[:, 0]
        py = position[:, 1]
        plt.plot(px, py)
        plt.scatter(px, py)

        plt.title('Test case')
        plt.xlabel(r'$\bf{x}$')
        plt.ylabel(r'$\bf{y}$')
        plt.xticks(())
        plt.yticks(())
        plt.colorbar()
        plt.show()

    pass

start = np.array([2, 1])
bounds = [(-100, 100)] * 2
bounds = np.array(bounds)

opt = ScipyOptimizers(100, scaling_factor = [1, 1], pgtol = 10e-10)
opt.initialize(start, f, g, bounds, plot)
res = opt.run()
print(res)
plot(np.array(opt.params_hist))
