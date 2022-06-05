import numpy as np


class GaussianProcess:

    def kernelGA(x, y, theta):
        """
        Gaussian kernel:
            np.exp(-(x-y)**2/(2*theta**2))
        """
        return np.exp(-(y-x)**2/(2*theta**2))

    def kernelBR(x, y, theta):
        """
        Brownian kernel:
            np.minimum(x, y)*theta**2
        """
        return np.minimum(x, y)*theta**2

    def kernelOU(x, y, theta):
        """
        Ornstein-Uhlenbeck kernel:
            np.exp(-np.abs(y-x)/theta**2)
        """
        return np.exp(-np.abs(y-x)/theta**2)

    def kernelPE(x, y, theta):
        """
        Periodic kernel:
            np.exp(-(2*np.sin((y-x)/2)**2)/theta**2)
        """
        return np.exp(-(2*np.sin((y-x)/2)**2)/theta**2)

    def kernelSY(x, y, theta):
        """
        Symmetric kernel:
            np.exp(-np.minimum(np.abs(y-x), np.abs(x+y))**2/theta**2)
        """
        return np.exp(-np.minimum(np.abs(y-x), np.abs(x+y))**2/theta**2)

    def build_randfunc(x, hyperparam, kernel, nb_points=100):
        """
        Main function that builds the gaussian process.
        Kernels available:
            - Gaussian (kernelGA)
            - Brownian (kernelBR)
            - Ornstein-Uhlenbeck (kernelOU)
            - Periodic (kernelPE)
            - Symmetric (kernelSY)
        """
        covar = np.zeros((x.shape[0], x.shape[0]))

        for idx in range(x.shape[0]):
            for jdx in range(x.shape[0]):
                covar[idx, jdx] = kernel(x[idx], x[jdx], hyperparam)

        U, D, V = np.linalg.svd(covar)
        diagD = np.diag(D)

        randvec = np.random.normal(size=x.shape)

        Y = np.dot(U, np.dot(np.sqrt(diagD), randvec))

        return Y
