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


###############################################################################

def spectral_clustering(
    graph_laplacian:GraphLaplacian
    )->dict:
    """ note,
    """

    fiedler_vector = graph_laplacian.eigvec[:,1]

    # note unsupervised learning sign assigned +/- randomly ...
    spectral_clustering_classifier = np.sign(fiedler_vector)

    misclassified_count, spectral_clustering_accuracy = evaluate_classification(
        classifier = spectral_clustering_classifier,
        data = graph_laplacian.data
        )

    indx = np.argmin(misclassified_count)  # {0,1}

    return {
        'laplacian':graph_laplacian,
        'fiedler':fiedler_vector * (-1)**indx,
        'classifier':spectral_clustering_classifier * (-1)**indx,
        'accuracy':spectral_clustering_accuracy[indx],
        'misclassified':misclassified_count[indx]
        }



def semisupervised_learning(
    graph_laplacian:GraphLaplacian,
    sample_size:int, # rows
    eigen_depth:int  # cols
    )->dict:
    """
    """
    laplacian_embedding = graph_laplacian.eigvec[:sample_size,:eigen_depth]
    sample_set = graph_laplacian.data[:sample_size]

    model = np.linalg.lstsq(
        a = laplacian_embedding,
        b = sample_set,
        rcond = None
        )[0]

    regression_predictor = graph_laplacian.eigvec[:,:eigen_depth].dot(model)
    ## predictor row wise max all else -1
    regression_classifier = np.sign(regression_predictor)

    misclassified_count, regression_accuracy = evaluate_classification(
        classifier = regression_classifier,
        data = graph_laplacian.data
        )

    indx = np.argmin(misclassified_count)  # {0,1}

    return {
        'linear_model':model,
        'predictor':regression_predictor,
        'classifier':regression_classifier * (-1)**indx,
        'accuracy':regression_accuracy[indx],
        'misclassified':misclassified_count[indx]
        }


def evaluate_classification(
    classifier:np.ndarray,
    data:np.ndarray
    )->tuple:
    """
    """
    misclassified = -1*np.ones(2)
    accuracy = -1*np.ones(2)

    # -1**0 = 1, index 0 -> sign of +1 corresponding to inequality for misclassification
    misclassified[0] = np.not_equal(data, classifier).sum()
    misclassified[1] = np.equal(data, classifier).sum()
    accuracy[0] = 1 - 1/data.size * misclassified[0]
    accuracy[1] = 1 - 1/data.size * misclassified[1]

    return misclassified, accuracy



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
