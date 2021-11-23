# Import the necessary modules/libraries
import numpy as np
from numpy.polynomial.polynomial import Polynomial

class EuclideanConfigurationSet:
    """
    Euclidean Configuration Set class
    """
    def __init__(self, confs):
        """
        Create a EuclideanConfigurationSet object corresponding to confs
        :param confs: configurations as an Nxnx3 or an nx3 numpy array
        """
        if self.is_valid(confs):
            self.confs = confs.copy()
            self.N, self.n, _ = self.confs.shape


    def is_valid(self, confs):
        """
        Check if self.conf is a valid nx3 numpy array
        :return: bool
        """
        if isinstance(confs, np.ndarray):
            return (len(confs.shape) == 3) & (confs.shape[-1] == 3)


    def AS_dets(self): #TODO: make this method able to handle confs with vertical directions
        """
        Return the AS det of the EuclideanConfiguration object
        :return: float
        """
        # compute j of a vector in C2
        def quatj(u):
            return [-u[1].conjugate(), u[0].conjugate()]

        # Compute the Hopf lifts of all pairwise directions
        def hopf_lifts():
            hl = np.zeros((self.N, self.n, self.n, 2), dtype=np.complex)
            norms = np.zeros((self.N, self.n, self.n), dtype=np.float)
            for i in range(self.n - 1):
                for j in range(i+1, self.n):
                    xs_ij = -self.confs[:, i, :] + self.confs[:, j, :]
                    rs_ij = np.linalg.norm(xs_ij, axis=-1)
                    for k in range(self.N):
                        if xs_ij[k, 2] != rs_ij[k]:
                            hopf_lift = [1., (xs_ij[k, 0] + 1.j*xs_ij[k, 1]) / (rs_ij[k] - xs_ij[k, 2])]
                            hl[k, i, j, :] = hopf_lift
                            norms[k, i, j] = np.linalg.norm(hopf_lift)
                        elif xs_ij[k, 2] == rs_ij[k]:
                            hl[k, i, j, :] = [0., -1.]
                            norms[k, i, j] = 1.
                        hl[k, j, i, :] = quatj(hl[k, i, j, :])
            return hl, norms

        dets = np.zeros((self.N,), dtype=np.complex)

        # Calculate the Hopf lifts of self.conf
        hl, norms = hopf_lifts()

        # loop over the set of configurations self.confs
        for k in range(self.N):
            # Populate the nxn matrix containing the coefficients of the AS polys
            M = np.zeros((self.n, self.n), dtype=np.complex)

            for i in range(self.n):
                roots = []
                for j in range(self.n):
                    if j != i:
                        roots.append(hl[k, i, j, 1]/hl[k, i, j, 0])
                poly = np.polynomial.polynomial.polyfromroots(roots)
                M[i, 0:poly.size] = poly

            num_k = np.linalg.det(M)

            factor_k = 1.
            for i in range(self.n - 1):
                for j in range(i+1, self.n):
                    factor_k *= hl[k, i, j, 0]
                    factor_k *= -hl[k, i, j, 1].conjugate()

            den_k = 1.
            for i in range(self.n - 1):
                for j in range(i+1, self.n):
                    den_k *= norms[k, i, j]**2

            dets[k] = num_k * factor_k / den_k

        return dets


class EuclideanConfiguration(EuclideanConfigurationSet):
    """
    Euclidean Configuration class
    """
    def __init__(self, conf):
        """
        Create a EuclideanConfiguration object corresponding to conf
        :param conf: configurations as an nx3 numpy array
        """
        super().__init__(np.expand_dims(conf.copy(), 0))


    def AS_det(self):
        """
        Compute the AS det of the EuclideanConfiguration object
        :return: complex
        """
        return super().AS_dets()[0]