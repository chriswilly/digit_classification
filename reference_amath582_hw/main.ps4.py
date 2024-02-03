"""
Michael Willy
AMATH582 Homework 4
March 3, 2022
"""
import logging
import datetime
import pathlib
import numpy as np
import pandas as pd
import scipy.spatial
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools

from sys import stdout
from dataclasses import (
    dataclass,
    field
    )

# from sklearn.metrics import mean_square_error
# import sklearn as skl
# import sklearn.model_selection
# import sklearn.kernel_ridge
from sklearn import linear_model

from IPython import embed

np.set_printoptions(precision=3, threshold=13)
pd.set_option('display.precision', 3)


def start_log()->pathlib.Path:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file = pathlib.Path(__file__).resolve()
    log_name = f'./logs/{file.parent.name}_{file.stem}_{timestamp}.log'
    log_file = pathlib.Path(log_name).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO
        )
    handler = logging.StreamHandler(stdout)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)

    return logger, log_file


@dataclass
class Dataset:
    """
    """
    category: str
    x: pd.DataFrame
    y: pd.Series = None


@dataclass
class GraphLaplacian:
    """
    """
    data: Dataset
    distance_ratio: float

    length_scale: float   = field(init=False)
    distance: np.ndarray  = field(init=False, repr=False)
    weights: np.ndarray   = field(init=False, repr=False)
    laplacian: np.ndarray = field(init=False, repr=False)
    eigval: np.ndarray    = field(init=False, repr=False)
    eigvec: np.ndarray    = field(init=False, repr=False)


    @staticmethod
    def gaussian_weight(t:np.ndarray, # 2D
                        sigma:float,
                        )->np.ndarray:
        """ gaussian centered about origin
        """
        return np.exp(-1/(2*sigma**2) * t**2)


    def __post_init__(self):
        """ compute eigen decomposition on weighted distance matrix
        """
        self.distance = scipy.spatial.distance_matrix(self.data.x, self.data.x)

        self.length_scale = self.distance_ratio*self.distance.mean()
        # be careful about t**2 
        self.weights = self.gaussian_weight(
            t = self.distance,
            sigma = self.length_scale
            )
        # L = D - W
        # unnormalized
        self.laplacian = -self.weights + np.diag(self.weights.sum(axis=1))
        
        # locally scoped working variables
        eigval_unsorted, eigvec_unsorted = np.linalg.eigh(self.laplacian)
        indx = eigval_unsorted.argsort()
        
        # attribues:
        self.eigval = eigval_unsorted[indx]
        self.eigvec = eigvec_unsorted[:,indx]



def load_data(file:str,**kwargs)->pd.DataFrame:
    """ load csv files into pandas dataframe
        optional labels object
    """

    column_labels = kwargs.get('labels')
    # redundant type cast if input as path obj
    file_path = pathlib.Path(file).resolve()

    if not file_path.exists():
        raise ValueError(f'{file}\nNot found.')

    if column_labels is not None:
        if (isinstance(column_labels,pd.DataFrame) 
            or isinstance(column_labels,pd.Series)):
            column_labels_list = column_labels.squeeze().tolist()
        
        elif (isinstance(column_labels,list) 
            or isinstance(column_labels,tuple) 
            or isinstance(column_labels,np.ndarray)):
            column_labels_list = list(column_labels)

        else:
            raise ValueError(f'column labels type {type(column_labels)}')
        return pd.read_csv(
            file_path, 
            names=column_labels_list
            )
    else:
        return pd.read_csv(file_path,header=None)


def mpl_params():
    """ matplotlib parameters plot formatting
    """
    mpl.rcParams['figure.titlesize'] = 44
    mpl.rcParams['figure.titleweight'] = 'demibold'
    mpl.rcParams['axes.labelsize'] = 38
    mpl.rcParams['axes.titlesize'] = 40
    mpl.rcParams['xtick.labelsize'] = 36
    mpl.rcParams['ytick.labelsize'] = 36
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 26
    mpl.rcParams['lines.markeredgewidth'] = 2
    mpl.rcParams['legend.framealpha'] = 0.87
    mpl.rcParams['legend.fontsize'] = 36


def scatter_plot(
    data:Dataset,
    file_name:str=None,
    labels:tuple = ('',''),
    ext:str='.png',
    **kwargs
    )->None:
    """
    """
    if not file_name:
        file_name = data.category

    output = pathlib.Path('plots').joinpath(file_name).with_suffix(ext)
    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(111)

    sigma = kwargs.get('arbitrary_sigma',False)

    if sigma:
        predictor_label = r'Predicton with $\sigma \approx %0.2f$'%sigma
    else:
        predictor_label = 'Predicton'


    ax.plot(data.y,'r.',label='True Values')
    ax.plot(data.x,'b.',label=predictor_label)


    limits = [
        0, data.y.size, 
        np.floor(data.y.min()), 
        np.ceil(data.y.max())
        ]

    plt.axis(limits)
    plt.title(f'{file_name}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    ax.legend(loc=0)

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[1]-limits[0])/4))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[3]-limits[2])/4))

    plt.grid(visible=True, which='major', axis='both')
    fig.savefig(output, bbox_inches='tight')
    plt.close('all')


def line_plot(
    data:Dataset,
    file_name:str=None,
    labels:tuple = ('',''),
    ext:str='.png',
    **kwargs
    )->None:
    """
    """
    if not file_name:
        file_name = data.category

    output = pathlib.Path('plots').joinpath(file_name).with_suffix(ext)
    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(111)

    ax.plot(data.x, data.y,'k-',label=f'Accuracy Profile')
    
    indx = kwargs.get('optimal_index',False)

    if indx:
        ax.plot([data.x[indx],data.x[indx]],[0,data.y[indx]],'r--',label=r'Optimal $\sigma\approx %0.2f$'%np.round(data.x[indx],2))
        ax.plot(data.x[indx],data.y[indx],'r.')


    limits = [
        np.floor(data.x.min()), 
        np.ceil(data.x.max()), 
        np.floor(data.y.min()), 
        np.ceil(data.y.max())
        ]

    plt.axis(limits)
    plt.title(f'{file_name}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    ax.legend(loc=0)

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[1]-limits[0])/4))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[3]-limits[2])/4))

    plt.grid(visible=True, which='major', axis='both')
    fig.savefig(output, bbox_inches='tight')
    plt.close('all')


def eigval_plot(
    data:Dataset,
    file_name:str=None,
    labels:tuple = ('',''),
    marker:str='k.',
    ext:str='.png',
    **kwargs
    )->None:
    """
    """
    if not file_name:
        file_name = data.category

    output = pathlib.Path('plots').joinpath(file_name).with_suffix(ext)
    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(111)

    sigma = kwargs.get('arbitrary_sigma',False)

    if sigma:
        y_label = r'$\lambda$ for $\sigma \approx$ %0.2f'%sigma
    else:
        y_label = 'Eigenvalues'


    ax.plot(data.y,marker,label=y_label)
    padding = 0.1
    limits = [
        0-padding, data.y.size, 
        np.floor(data.y.min()-padding), 
        np.ceil(data.y.max()+padding)
        ]

    plt.axis(limits)
    plt.title(f'{file_name}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    ax.legend(loc=0)

    # ax.set_yscale('log')
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[1]-limits[0])/4))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[3]-limits[2])/4))

    plt.grid(visible=True, which='major', axis='both')
    fig.savefig(output, bbox_inches='tight')
    plt.close('all')


def log_data_info(
    logger:logging.Logger,
    data:pd.DataFrame,
    name:str=None
    )->None:
    """ handles repeated verbose log string
    """
    logger.info(f'\n\tData loaded {name}: \n\tShape {data.shape}\n\tType\n{data.dtypes}\n\tMean\n{data.mean(axis=0)}\n\tStdev\n{data.std(axis=0)}')




def spectral_clustering(train:Dataset,distance_ratio:float)->dict:
    """ note,
    """

    graph_laplacian = GraphLaplacian(
        data = train,
        distance_ratio = distance_ratio
        )

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
    sample_set = graph_laplacian.data.y.iloc[:sample_size]

    model = np.linalg.lstsq(
        a=laplacian_embedding,
        b=sample_set
        ,rcond=None)[0]

    regression_predictor = graph_laplacian.eigvec[:,:eigen_depth].dot(model)

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
    data:Dataset
    )->tuple:
    """ 
    """
    misclassified = -1*np.ones(2)
    accuracy = -1*np.ones(2)

    # -1**0 = 1, index 0 -> sign of +1 corresponding to inequality for misclassification
    misclassified[0] = np.not_equal(data.y, classifier).sum()
    misclassified[1] = np.equal(data.y, classifier).sum()
    accuracy[0] = 1 - 1/data.y.size * misclassified[0]
    accuracy[1] = 1 - 1/data.y.size * misclassified[1]

    return misclassified, accuracy




def main(data_root:pathlib.Path = pathlib.Path(__file__).resolve().parent)->None:
    """
    """
    # helper functions
    logger,_ = start_log()
    mpl_params()

    # Load and package data into dataclass
    train_df = load_data(
        file = data_root.joinpath('data/house-votes-84.data'),
        labels = load_data(
            file=data_root.joinpath('data/titles.csv')
            )
        )

    # construct dataset class with _.x _.y attr notation for convenience
    # cast to numerical y->+1 n->-1 ?->0
    train = Dataset(
        category = 'Train',
        x = (train_df.iloc[:,1:]
            .replace('n',-1)
            .replace('y', 1)
            .replace('?', 0)
            .astype(int)
            ),
        y = (train_df.iloc[:,0]
            .replace('republican',-1)
            .replace('democrat', 1)
            .astype(int)
            )
        )

    problem_two = spectral_clustering(
        train=train,
        distance_ratio=0.75)

    misclassified_count = problem_two['misclassified']
    spectral_clustering_accuracy = problem_two['accuracy']
    spectral_clustering_classifier = problem_two['classifier']
    fiedler_vector = problem_two['fiedler']
    graph_laplacian = problem_two['laplacian']

    logger.info('part 2 a,b')
    logger.info(f'{list(graph_laplacian.__dict__.keys())}')
    logger.info(f'sigma {np.round(graph_laplacian.length_scale,3)}')
    logger.info(f'parameter ratio input {np.round(graph_laplacian.distance_ratio,3)}')
    logger.info(f'misclassified {misclassified_count}')
    logger.info(f'accuracy {np.round(spectral_clustering_accuracy,3)}')
    logger.info('-------------------------------------------------------------')
    #---------------------------------------------------------------------------

    party_sort = train.y.sort_values().index

    spectral_inspect = Dataset(
        category = 'Spectral Clustering Fiedler Vector',
        x = fiedler_vector[party_sort],
        y = train.y.iloc[party_sort].reset_index(drop=True)
        )

    scatter_plot(data=spectral_inspect,labels=('Index (Sorted)','Raw Value'),
        arbitrary_sigma=graph_laplacian.length_scale)
    
    spectral_comparison = Dataset(
        category = 'Spectral Clustering Predictor',
        x = spectral_clustering_classifier[party_sort],
        y = train.y.iloc[party_sort].reset_index(drop=True)
        )
    # scatter_plot(data=spectral_comparison,labels=('Index (Sorted)','Value'))

    #---------------------------------------------------------------------------
    # distance ratio is a ratio to distance mean, cannot fix this now roll with it to find parameter sigma = distance ratio / mean
    parameter_range = np.linspace(1e-6,4,100)
    normalizing_factor = graph_laplacian.distance.mean()

    # init
    accuracy_profile = np.zeros(parameter_range.size)
    misclassified_profile = np.zeros(parameter_range.size)

    for indx,sigma in enumerate(parameter_range):

        param = sigma / normalizing_factor
        temp_dict = spectral_clustering(train=train,distance_ratio=param)
        
        accuracy_profile[indx] = temp_dict['accuracy']
        misclassified_profile[indx] = temp_dict['misclassified']

    indx_optimal = np.argmax(accuracy_profile)
    # this looks so dumb but it's making sense....
    sigma_optimal = parameter_range[indx_optimal]
    
    param_optimal = sigma_optimal / normalizing_factor

    parameter_comparison = Dataset(
        category = 'Spectral Clustering Parameter Sweep',
        x = parameter_range,
        y = accuracy_profile
        )

    line_plot(data=parameter_comparison,labels=('Distance Parameter','Computed Accuracy'),optimal_index=indx_optimal)

    logger.info('part 2 c')
    logger.info(f'sigma {np.round(sigma_optimal,3)}')
    logger.info(f'parameter ratio input {np.round(param_optimal,3)}')
    logger.info(f'misclassified {misclassified_profile[indx_optimal]}')
    logger.info(f'accuracy {np.round(accuracy_profile[indx_optimal],3)}')
    logger.info('-------------------------------------------------------------')


    #---------------------------------------------------------------------------
    logger.info('part 3')

    problem_three = spectral_clustering(train=train,distance_ratio=param_optimal)
    graph_laplacian = problem_three['laplacian']

    column_range = np.arange(2,7)
    row_range = np.array([5,10,20,40])

    regression_accuracy = pd.DataFrame(
        np.zeros((row_range.size*column_range.size,3)),
        ).astype({0:int, 1:int, 2:float})

    regression_accuracy.columns = ['Eigenvectors M','Training Samples J', 'Accuracy']


    indx = 0
    for sample_size,eigen_depth in itertools.product(row_range,column_range):

        temp_dict= semisupervised_learning(
            graph_laplacian = graph_laplacian,
            sample_size = sample_size,
            eigen_depth = eigen_depth
            )
        regression_accuracy.iloc[indx,:] = (
            eigen_depth,
            sample_size,
            temp_dict['accuracy']
            )
        indx+=1



    regression_inspect = Dataset(
        category = 'Laplacian Regression',
        x = temp_dict['predictor'][party_sort],
        y = train.y.iloc[party_sort].reset_index(drop=True)
        )

    scatter_plot(data=regression_inspect,labels=('Index (Sorted)','Raw Value'),
        arbitrary_sigma = graph_laplacian.length_scale
        )

    eigval_logs = Dataset(
        category = 'Graph Laplacian Eigvals',
        x = None,
        y = np.log(graph_laplacian.eigval[1:])
        )
    eigval_delta = Dataset(
        category = 'Delta Graph Laplacian Eigvals',
        x = None,
        y = 100*np.diff(graph_laplacian.eigval)[:32]
        )

    eigval_plot(data=eigval_logs,labels=('Index (Sorted)',r'log($\lambda$)'),marker='k.',
        arbitrary_sigma = graph_laplacian.length_scale
        )

    eigval_plot(data=eigval_delta,labels=('Index (Sorted)',r'$\Delta\lambda\times100$'),marker='k-',
        arbitrary_sigma = graph_laplacian.length_scale
        )

    logger.info(regression_accuracy.pivot_table(
        'Accuracy',
        index='Eigenvectors M',
        columns='Training Samples J'
        )
    )

if __name__=='__main__':
    main()