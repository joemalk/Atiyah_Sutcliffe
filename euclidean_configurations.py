# Import the necessary modules/libraries
import numpy as np
from scipy.linalg import expm

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
        Check if self.conf is a valid Nxnx3 numpy array
        :return: bool
        """
        if isinstance(confs, np.ndarray):
            return (len(confs.shape) == 3) & (confs.shape[-1] == 3)


    def has_distinct_points(self):
        """
        Check if all points within a configuration are distinct
        :return: bool
        """
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                xs_ij = -self.confs[:, i, :] + self.confs[:, j, :]
                xs_ij.reshape(self.N,3)
                if (xs_ij == [0., 0., 0.]).all(axis=-1).any():
                    return False
        return True


    def AS_dets(self):
        """
        Return the AS det of the EuclideanConfiguration object
        :return: float
        """
        if not self.has_distinct_points():
            raise Exception("Error: one of the configurations does not have distinct points.")

        confs = self.confs.copy()

        def has_vertical_directions(confs):
            """
            Check for vertical directions (parallel to the z-axis)
            :return: bool
            """
            for i in range(self.n - 1):
                for j in range(i + 1, self.n):
                    xs_ij = -confs[:, i, :] + confs[:, j, :]
                    if (xs_ij[:, :2] == [0., 0.]).all(axis=-1).any():
                        return True
            return False

        while has_vertical_directions(confs):
            eps = np.random.random()
            inf_rot = np.array([[0.,0.,-1.],[0.,0.,0.],[1.,0.,0.]])
            rot = expm(eps * inf_rot)
            confs = confs.dot(rot)


        # compute j of a vector in C2
        def quatj(u):
            return np.concatenate([-u[:,1:2].conjugate(), u[:,0:1].conjugate()], axis= -1)


        # Compute the Hopf lifts of all pairwise directions
        def hopf_lifts(confs):
            hl = np.zeros((self.N, self.n, self.n - 1, 2), dtype=np.complex)

            for i in range(self.n - 1):
                for j in range(i+1, self.n):
                    xs_ij = -confs[:, i, :] + confs[:, j, :]
                    rs_ij = np.linalg.norm(xs_ij, axis=-1).reshape(-1,1)
                    if (xs_ij[:, 2] != rs_ij[:]).all():
                        hopf_lift = np.concatenate([np.ones((self.N,1), dtype=complex), \
                                                    (xs_ij[:, 0:1] + 1.j * xs_ij[:, 1:2]) / (rs_ij - xs_ij[:, 2:3])],\
                                                   axis=-1)
                        hl[:, i, j-1, :] = hopf_lift / np.linalg.norm(hopf_lift, axis=-1).reshape(-1,1)
                    else:
                        raise Exception("Error: division by 0 due to a vertical direction!")
                    hl[:, j, i, :] = quatj(hl[:, i, j-1, :])
            return hl

        # Calculate the Hopf lifts of confs
        hl = hopf_lifts(confs)

        def poly_mul(polys_arr, linear_arr):
            N1,n1,d1 = polys_arr.shape
            N2,n2,d2 = linear_arr.shape
            assert(N1 == self.N)
            assert(n1 == self.n)
            assert(N2 == self.N)
            assert(n2 == self.n)
            assert(d2 == 2)
            prod = np.zeros((self.N, self.n, d1+1), dtype=complex)
            prod[:,:,:-1] = linear_arr[:,:,0:1]*polys_arr
            prod[:,:,1:] += linear_arr[:,:,1:2]*polys_arr
            return prod

        polys = np.ones((self.N, self.n, 1), dtype=complex)
        for j in range(self.n-1):
            polys = poly_mul(polys, hl[:,:,j,:])

        return np.linalg.det(polys)


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