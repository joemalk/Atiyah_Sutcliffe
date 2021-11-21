# Import the necessary modules/libraries
import numpy as np
from numpy import poly1d

class EuclideanConfiguration:
    """
    Euclidean Configuration class
    """
    def __init__(self, conf):
        """
        Create a EuclideanConfiguration object corresponding to conf
        :param conf: configuration as an nx3 numpy array
        """
        self.conf = conf
        if self.is_valid():
            self.n = conf.shape[0]


    def is_valid(self):
        """
        Check if self.conf is a valid nx3 numpy array
        :return: bool
        """
        if isinstance(self.conf, np.ndarray):
            if len(self.conf.shape) == 2:
                if self.conf.shape[1] == 3:
                    return True
        return False

    def AS_det(self):
        """
        Return the AS det of the EuclideanConfiguration object
        :return: float
        """
        # compute j of a vector in C2
        def quatj(u):
            if u.size != 2:
                print("Error, the vector is not in C2")
                return
            if np.linalg.norm(u) == 0:
                print("Error, the vector in C2 is 0")
                return
            return np.array([-u[1].conjugate(), u[0].conjugate()])

        # Compute the stereographic projection of a non-zero vector in R3
        def stereo(x):
            if x.size != 3:
                print("Error, the vector is not in R3")
                return
            r = np.linalg.norm(x)
            if r == 0:
                print("Error, the vector in R3 is 0")
                return
            if x[2] != r:
                return np.array([1, complex(x[0], x[1]) / (r - x[2])])
            elif x[2] == r:
                return np.array([0., -1.])

        # Calculate the Hopf lifts of self.conf
        p = np.zeros((self.n, self.n, 2), dtype=np.complex)
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                p[i, j, :] = stereo(-self.conf[i, :] + self.conf[j, :])
                p[j, i, :] = quatj(p[i, j, :])

        # Populate the nxn matrix containing the coefficients of the AS polys
        M = np.zeros((self.n, self.n), dtype=np.complex)
        for i in range(self.n):
            mypoly = poly1d([complex(1, 0)], variable="t")
            for j in range(self.n):
                if j != i:
                    mypoly = mypoly * poly1d(np.array([p[i,j,0],-p[i,j,1]], dtype=np.complex))
            mycoeffs = mypoly.coeffs[::-1]
            M[i, 0:mycoeffs.size] = mycoeffs[:]

        # Calculate the denominator of the AS det
        P = 1.
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                P_ij = np.concatenate([p[i,j].reshape(1,-1), p[j,i].reshape(1,-1)], axis=0)
                P *= np.linalg.det(P_ij)

        return np.linalg.det(M)/P


