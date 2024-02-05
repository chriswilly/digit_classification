"""

"""
import numpy as np
import numpy.matlib
import scipy.spatial
from dataclasses import (
    dataclass,
    field
    )
# import IPython

np.set_printoptions(precision=1, threshold=80)


@dataclass
class GraphLaplacian:
    """
    """
    data: np.ndarray      = field(init=True, repr=False)
    distance_ratio: float = field(init=True)

    # sigma or radial threshold
    length_scale: float   = field(init=False)
    distance:  np.ndarray = field(init=False, repr=False)
    weights:   np.ndarray = field(init=False, repr=False)
    
    laplacian: np.ndarray = field(init=False, repr=False)
    eigval:    np.ndarray = field(init=False, repr=False)
    eigvec:    np.ndarray = field(init=False, repr=False)
    # L_2 norm:
    laplacian_norm: np.ndarray = field(init=False, repr=False)
    eigval_norm:    np.ndarray = field(init=False, repr=False)
    eigvec_norm:    np.ndarray = field(init=False, repr=False)


    @staticmethod
    def gaussian_weight(
        t:np.ndarray, # 2D
        sigma:float,
        )->np.ndarray:
        """ 
        gaussian centered about origin
        """
        return np.exp(-1/(2*sigma**2) * t**2)


    @staticmethod
    def radius(
        t:np.ndarray, # 2D
        sigma:float,
        )->np.ndarray:
        """
        radius cutoff
        """
        filtered = np.zeros(t.shape)
        indx = t<=sigma
        filtered[indx] = t[indx]

        return filtered


    def __post_init__(self):
        """
        compute eigen decomposition on weighted distance matrix
        """
        self.distance = scipy.spatial.distance_matrix(self.data, self.data)

        self.length_scale = self.distance_ratio * self.distance.mean()

        self.weights = self.gaussian_weight(
            t = self.distance,
            sigma = self.length_scale
            )

        # L = D - W
        # unnormalized
        self.laplacian = -self.weights + np.diag(self.weights.sum(axis=1))
        self.laplacian_norm = (
            self.laplacian / np.linalg.norm(self.laplacian, keepdims=True)
            )
            # keepdims only required in higher dims >=3 for p=2
        
        # locally scoped working variables, note eigh is for symmetric array which we have from distance ||u-u||
        eigval_unsorted, eigvec_unsorted = np.linalg.eigh(self.laplacian)
        indx = eigval_unsorted.argsort()

        eigval_norm_unsorted, eigvec_norm_unsorted = np.linalg.eigh(self.laplacian_norm)
        indy = eigval_norm_unsorted.argsort()

        # attribues:
        self.eigval = eigval_unsorted[indx]
        self.eigvec = eigvec_unsorted[:,indx]
        
        self.eigval_norm = eigval_norm_unsorted[indy]
        self.eigvec_norm = eigvec_norm_unsorted[:,indy]


# def test():
#     """
#     """
#     raw = np.zeros([9,9])
#     raw[:3,:3] = -1
#     raw[-3:,-3:] = 1
#     raw[-3:,:3] = 2
#     resolution = 0.1
#     # rng = np.random.default_rng()

#     test = GraphLaplacian(
#         data=raw,
#         distance_ratio=resolution
#         )
#     indx = (test.eigval<=5e-5)
#     print(f'{np.sum(indx)} distict groups given {resolution} relative distance')
#     IPython.embed()


if __name__=='__main__':
    test()
