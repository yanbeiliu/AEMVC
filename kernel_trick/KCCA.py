import numpy
import scipy.linalg
from kernel_trick.kernel.AbstractKernel import AbstractKernel

"""
An implementation of the Kernel Canonincal Correlation Analysis (KCCA) algorithm. 
"""


class KernelCCA(object):
    def __init__(self, tau):

        self.tau = tau

    def learnModel(self, Kx, Ky):
        """
        Learn the KCCA  directions using set of examples given the numpy.ndarrays
        X and Y. If X and Y are matrices then their rows are examples, and they must
        have the same number of rows.

        :param X: The X examples.
        :type X: :class:`numpy.ndarray`

        :param Y: The Y examples.
        :type Y: :class:`numpy.ndarray`

        :returns alpha: The dual directions in the X space.
        :returns beta: The dual directions in the Y space.
        :returns lambda: The correlations for each projected dimension.
        """

        Kxx = numpy.dot(Kx, Kx)
        Kyy = numpy.dot(Ky, Ky)

        numExamples = Kx.shape[0]

        Z = numpy.zeros((numExamples, numExamples))

        A = numpy.c_[Z, numpy.dot(Kx, Ky)]
        A1 = numpy.c_[numpy.dot(Ky, Kx), Z]
        A = numpy.r_[A, A1]

        B = numpy.c_[(1 - self.tau) * Kxx - self.tau * Kx, Z]
        B1 = numpy.c_[Z, (1 - self.tau) * Kyy - self.tau * Ky]
        B = numpy.r_[B, B1]

        A = ((A + A.T) / 2).astype(numpy.float32)
        B = ((B + B.T) / 2).astype(numpy.float32)

        (D, W) = scipy.linalg.eig(A, B)

        # Only select eigenvalues which are greater than zero
        W = W[:, D > 0]

        # We need to return those eigenvectors corresponding to positive eigenvalues
        self.alpha = W[0:numExamples, :]
        self.beta = W[numExamples:numExamples * 2, :]
        self.lmbdas = D[D > 0]

        alphaDiag = self.alpha.T.dot(Kxx).dot(self.alpha)
        alphaDiag = alphaDiag + numpy.array(alphaDiag < 0, numpy.int)
        betaDiag = self.beta.T.dot(Kyy).dot(self.beta)
        betaDiag = betaDiag + numpy.array(betaDiag < 0, numpy.int)
        self.alpha = numpy.dot(self.alpha, numpy.diag(1 / numpy.sqrt(numpy.diag(alphaDiag))))
        self.beta = numpy.dot(self.beta, numpy.diag(1 / numpy.sqrt(numpy.diag(betaDiag))))

        return self.alpha, self.beta, self.lmbdas

    def project(self, testX, testY, k=None):
        """
        Project the examples in the KCCA subspace using set of test examples testX
        and testY. The number of projection directions is specified with k, and
        if this parameter is None then all directions are used.

        :param testX: The X examples to project.
        :type testX: :class:`numpy.ndarray`

        :param testY: The Y examples to project.
        :type testY: :class:`numpy.ndarray`

        :returns testXp: The projections of testX.
        :returns testYp: The projections of testY.
        """
        if k == None:
            k = self.alpha.shape[1]

        return numpy.dot(testX, abs(self.alpha[:, 0:k])), numpy.dot(testY, abs(self.beta[:, 0:k]))
