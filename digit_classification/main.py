"""
Michael Willy
AMATH582 Homework 2 + 4 combined digit classifier
January 29, 2022 update February 2, 2024
"""
import argparse
import logging
import datetime
import pathlib
import numpy as np

from rich import print
from sys import stdout
from dataclasses import (
    dataclass,
    field
    )

import scipy.spatial
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from sklearn.decomposition import PCA
from sklearn import linear_model

from sklearn.manifold import SpectralEmbedding

# custom but write this out
from graph_laplacian import GraphLaplacian

import IPython
np.set_printoptions(precision=2, threshold=27)



def start_log(
    name:str,
    destination:str|pathlib.Path,
    caller:str = __name__,
    ext:str = '.log',
    level:object = logging.INFO
    )->tuple[logging.Logger,pathlib.Path]:
    """
    create log directory, return logger obj and path
    """
    timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S')

    if isinstance(destination,str):
        log_name = f'{destination}/{name}_{timestamp}'

    elif isinstance(destination,pathlib.Path):
        log_name = str(destination/f'{name}_{timestamp}')

    else:
        raise TypeError(f'{type(destination)=} not supported')

    log_file = pathlib.Path(log_name).with_suffix(ext).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=level,
        encoding='utf-16'
        )

    logger = logging.getLogger(caller)
    handler = logging.StreamHandler(sys.stdout)
    logging_formatter = logging.Formatter(r'%(asctime)s:%(name)s:%(message)s')
    handler.setFormatter(logging_formatter)
    logger.addHandler(handler)

    return logger, log_file


@dataclass
class ImageData:
    category: str
    x: np.ndarray
    y: np.ndarray = None
    binary_key: np.ndarray = None


def mpl_params()->None:
    """ matplotlib parameters plot formatting
    """
    mpl.rcParams['figure.titlesize'] = 32
    mpl.rcParams['figure.titleweight'] = 'demibold'
    mpl.rcParams['axes.labelsize'] = 32
    mpl.rcParams['axes.titlesize'] = 28
    mpl.rcParams['xtick.labelsize'] = 32
    mpl.rcParams['ytick.labelsize'] = 32
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    mpl.rcParams['lines.linewidth'] = 4
    mpl.rcParams['lines.markersize'] = 16
    mpl.rcParams['lines.markeredgewidth'] = 2
    mpl.rcParams['legend.framealpha'] = 0.87
    mpl.rcParams['legend.fontsize'] = 36

    cyc_color = cycler(color=['r','b','g','k','c','m','y'])
    cyc_lines = cycler(linestyle=['-', '--', ':'])
    cyc_alpha = cycler(alpha=[0.7, 0.35])
    cycles = (cyc_alpha * cyc_lines * cyc_color)
    mpl.rcParams['axes.prop_cycle'] = cycles


def plot_line(
    data:np.ndarray,  # 1D in singular vals
    file_name:str,
    ext:str='.png',
    labels:tuple = ('','')
    )->None:
    """
    """

    output = (
        pathlib.Path(kwargs.get('plot','plots'))
        .joinpath(file_name).with_suffix(ext)
        )

    output = pathlib.Path('plots').joinpath(f'{file_name[0]}.{file_name[-1]}')
    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(13.5,12))
    ax = fig.add_subplot(111)

    if len(data.shape)>1:
        ax.plot(data[...,0],data[...,1],'-k',label=f'')

    else:
        ax.plot(data,'-k',label=f'')


    plt.axis('tight')
    plt.title(f'{file_name[0]}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=data.shape[0]//4))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.abs(data).max()//4))

    plt.grid(visible=True, which='major', axis='both')
    # ax.legend(loc=0)

    # plt.show()
    fig.savefig(output) #,bbox_inches='tight')
    plt.close('all')



def plot_digits(
    data:ImageData,
    count:int,
    file_name:str,
    title:str = None,
    ext:str='.png'
    )->None:
    """ Plot NxN digits
    """

    if not title:
        try:
            title = f'First {count} {data.category.title()}ing Features'
        except:
            title = f'First {count} Features'

    output = (
        pathlib.Path(kwargs.get('plot','plots'))
        .joinpath(file_name).with_suffix(ext)
        )

    output.resolve().parent.mkdir(exist_ok=True)

    if count > data.x.shape[0]:
        raise ValueError(f'count of {count} exceeds {data.x.shape[0]} images')
    else:
        side_count = np.sqrt(count)

    if not side_count == int(side_count):
        raise ValueError(f'count {count} is not a perfect square')
    else:
        side_count = int(side_count) # cast to int

    pixel_length = int(np.sqrt(data.x.shape[1]))

    fig, ax = plt.subplots(side_count, side_count, figsize=(13.5,12))

    for k in range(count):
        indx = np.unravel_index(k, (side_count,side_count))
        ax[indx].imshow((data.x[k,:]
                        .reshape((pixel_length, pixel_length))),
                         cmap='Greys')
        ax[indx].axis('off')

    fig.suptitle(title)
    fig.savefig(output) #,bbox_inches='tight')
    plt.close('all')


def eigval_plot(
    data:np.array,
    file_name:str,
    labels:tuple = ('',''),
    marker:str='.',
    ext:str='.png',
    PAPER_SIZE:tuple=(11,9),
    **kwargs
    )->None:
    """
    """

    output = (
        pathlib.Path(kwargs.get('plot','plots'))
        .joinpath(file_name).with_suffix(ext)
        )

    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=PAPER_SIZE)
    ax = fig.add_subplot(111)

    sigma = kwargs.get('arbitrary_sigma',False)

    if sigma:
        y_label = fr'$\lambda$ for $\sigma \approx$ {sigma:0.2f}'
    else:
        y_label = 'Eigenvalues'


    ax.plot(data,marker,label=y_label)
    padding = 0.
    limits = [
        0-padding, data.size,
        np.floor(np.nanmin(data)-padding),
        np.ceil(np.nanmax(data)+padding)
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


def eigvec_3d_plot(
    graph:GraphLaplacian,
    file_name:str,
    depth:int = 3,
    labels:tuple = ('',''),
    marker:str='.',
    ext:str='.png',
    PAPER_SIZE:tuple=(9,9),
    **kwargs
    )->None:
    """
    """

    sigma = kwargs.get('arbitrary_sigma',False)

    for indx in np.arange(1,depth):

        if sigma:
            y_label = fr'$v$ for $\sigma \approx$ {sigma:0.2f}'

        else:
            y_label = f'Eigenvectors {indx},{indx+1}'

        fig = plt.figure(figsize=PAPER_SIZE)
        ax = fig.add_subplot(projection='3d')

        if not bool(np.mod(indx,2)):
            data_x = graph.eigvec_norm[:,indx+1]
            data_y = graph.eigvec_norm[:,indx]
            data_z = graph.eigvec_norm[:,indx+2]

            labels = (f'$v_{indx+1}$',f'$v_{indx}$'f'$v_{indx+2}$')
        else:
            data_x = graph.eigvec_norm[:,indx]
            data_y = graph.eigvec_norm[:,indx+1]
            data_z = graph.eigvec_norm[:,indx+2]

            labels = (f'$v_{indx}$',f'$v_{indx+1}$',f'$v_{indx+2}$')

        ax.scatter(data_x, data_y, data_z, marker=marker)

        plt.title(f'{file_name}')

        ax.legend(loc=0)

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        plt.show()


def eigvec_plot(
    graph:GraphLaplacian,
    file_name:str,
    depth:int = 3,
    labels:tuple = ('',''),
    marker:str='.',
    ext:str='.png',
    PAPER_SIZE:tuple=(11,9),
    **kwargs
    )->None:
    """
    """
    output = (
        pathlib.Path(kwargs.get('plot','plots'))
        .joinpath(file_name)
        )

    output.resolve().mkdir(parents=True, exist_ok=True)

    sigma = kwargs.get('arbitrary_sigma',False)


    for indx in np.arange(1,depth):

        if sigma:
            y_label = fr'$v$ for $\sigma \approx$ {sigma:0.2f}'

        else:
            y_label = f'Eigenvectors {indx},{indx+1}'

        fig = plt.figure(figsize=PAPER_SIZE)
        ax = fig.add_subplot(111)

        if not bool(np.mod(indx,2)):
            data_x = graph.eigvec_norm[:,indx+1]
            data_y = graph.eigvec_norm[:,indx]
            labels = (f'$v_{indx+1}$',f'$v_{indx}$')
        else:
            data_x = graph.eigvec_norm[:,indx]
            data_y = graph.eigvec_norm[:,indx+1]
            labels = (f'$v_{indx}$',f'$v_{indx+1}$')


        ax.plot(data_x,data_y,marker,label=y_label)
        padding = 0.0
        limits = [
            np.floor( data_x.min()-padding ),
            np.ceil(  data_x.max()+padding ),
            np.floor( data_y.min()-padding ),
            np.ceil(  data_y.max()+padding )
            ]

        plt.axis(limits)
        plt.title(f'{file_name}')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        ax.legend(loc=0)

        # ax.set_yscale('log')
        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.1f}'))
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[1]-limits[0])/4))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[3]-limits[2])/4))

        output = (
            pathlib.Path(kwargs.get('plot','plots'))
            .joinpath(f'{file_name}/{file_name}_{indx}')
            .with_suffix(ext)
            )

        plt.grid(visible=True, which='major', axis='both')
        fig.savefig(output, bbox_inches='tight')
        plt.close('all')


def find_PCA_count(
    model:PCA,
    ratio:float
    )->int:
    """
        iterate up thru full set of singular vals to see what number of PCA modes 
        will achieve ratio <1 of full set
    """
    for k in range(model.singular_values_.shape[0]):

        reference_value = ratio*np.sqrt(np.sum(model.singular_values_**2))

        test_value = np.sqrt(np.sum(model.singular_values_[:k]**2))
        
        if test_value >= reference_value:
            return k

    raise Exception(f'exceeded {model.singular_values_.shape[0]} modes,\
                     check your ratio value {ratio} as the threshold should be met')


## in case of: beta_train,_,_,_ = np.linalg.lstsq(a=A_train,b=train_subset.binary_key,rcond=None)

# def mean_square_error(
#     predictor:np.ndarray,
#     data:np.ndarray,
#     output:np.ndarray
#     )->float:
#     return 1/output.shape[0] * np.sum((data.dot(predictor) - output)**2)


def mean_square_error(
    prediction:np.ndarray,
    true_data:np.ndarray
    )->float:
    """
    """
    return 1/true_data.shape[0] * np.sum((prediction - true_data)**2)


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


def main(args:argparse.Namespace)->None:
    """
    """

    logger,log_file = start_log(
        name=f'{pathlib.Path(__file__).stem}',
        destination=args.log
        )

    mpl_params()

    # return nested dict inside scalar np ndarray obj
    # Load and package data into dataclass
    test_labels = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_test_labels.csv'),
        dtype=int
        )
    tain_labels = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_training_labels.csv'),
        dtype=int
        )
    test_features = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_test_features.csv'),
        dtype=float
        )
    train_features = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_training_features.csv'),
        dtype=float
        )

    train = ImageData(
        category = 'training digits',
        x = train_features,
        y = tain_labels
        )

    test  = ImageData(
        category = 'test digits',
        x = test_features,
        y = test_labels
        )

    logger.info(f'data loaded train: {train.x.shape}{train.y.shape}')
    logger.info(f'data loaded test:  {test.x.shape}{test.y.shape}')


    # init training model
    pca = PCA(n_components=train.x.shape[1])
    pca.fit(train.x)

    principal_components = ImageData(
        category='trained model',
        x = pca.components_
        )

    plot_digits(
        data=train,
        count=64,
        file_name='Training Features'
        )

    plot_digits(
        data=principal_components,
        count=16,
        file_name='Principal Components'
        )

    # inspect singular values
    plot_line(
        data = np.log(pca.singular_values_),
        file_name='Singular Values',
        labels = ('index','log(Singular Values)')
        )


    plot_line(
        data = 100 * pca.explained_variance_ratio_.cumsum(),
        file_name='Cumulative Explained Variance Ratio',
        labels = ('index','Cumulative Sum')
        )

    plot_line(
        data = 100 * pca.explained_variance_ratio_.cumsum()[:16],
        file_name='Cumulative Explained Variance Ratio Truncated',
        labels = ('index','Cumulative Sum')
        )

    print('loaded principal components for training')

    scale_factor = 0.05

    laplacian = GraphLaplacian(
        data=train.x,
        distance_ratio=scale_factor
        )

    indx = (laplacian.eigval<=5e-20)
    print(f'{np.sum(indx)} distict groups given {scale_factor} relative distance')


    ## checkpoint
    # IPython.embed()

    train.binary_key = np.zeros([train.y.shape[0],10],dtype=int)

    # lookup how to ravel & put vs this row by row loop
    for indx in np.arange(train.y.size):
        train.binary_key[indx,train.y[indx]-1] = 1

    # +1 / -1 one-hot instead of 1 / 0, this keeps symmetric about origin
    train.binary_key = train.binary_key*2 - 1


    eigval_plot(
        data = np.log(laplacian.eigval[1:]),
        file_name = 'Graph Laplacian Eigenvalues',
        labels = ('$Index_j$',r'$log(\lambda_j)$'),
        marker ='b.',
        arbitrary_sigma = laplacian.length_scale
        )


    eigval_plot(
        data = np.log(laplacian.eigval_norm[1:]),
        file_name = 'Graph Laplacian Eigenvalues Normalized',
        labels= ('$Index_j$',r'$log(\lambda_j)$'),
        marker='b.',
        arbitrary_sigma = laplacian.length_scale
        )


    eigvec_plot(
        graph=laplacian,
        file_name = 'Graph Laplacian Eigenvectors',
        depth=4,
        )

    eigvec_3d_plot(
        graph=laplacian,
        file_name = 'Graph Laplacian Eigenvectors',
        depth=2,
        )


    eigval_plot(
        data = 1000*np.diff(laplacian.eigval)[:8],
        file_name = 'Change in Eigenvalues',
        labels = ('$Index_m$',r'$\Delta\lambda$'),
        marker='k-',
        arbitrary_sigma = laplacian.length_scale
        )


    print('calc graph laplacian')
    IPython.embed()

    # estimate 
    thresholds = (0.6, 0.8, 0.9)

    # Explained variance approach - don't use lol

    [logger.info(f'{100*t}% index: {np.argmax(pca.explained_variance_ratio_.cumsum() >= t)}') for t in thresholds]

    # L2 / Frobenius norm of singular values
    pca_modes_threshold = {
        60:find_PCA_count(model=pca,ratio=0.6),
        80:find_PCA_count(model=pca,ratio=0.8),
        90:find_PCA_count(model=pca,ratio=0.9)
        }

    # print('pca_modes_threshold:\n',pca_modes_threshold)
    [logger.info(f'{key}% of L2 norm with {val} PCA modes') for key,val in pca_modes_threshold.items()]
    

    # part 3: 1|8
    indx = (train.y==1)|(train.y==8)

    train_subset = ImageData(
        category = '1,8 training subset',
        x = train.x[indx],
        y = train.y[indx]
        )

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==1] = -1
    train_subset.binary_key[train_subset.y==8] =  1

    # project X_1,8 onto first 16 PCA modes by dot product, 
    # PCA(n=16) is same as pca.components_[:16] so keep same model and just do the linalgebra if project -> dot project

    A_train = train_subset.x.dot(pca.components_[:16].transpose())
    # 455,256 * 256,16  -> 455 row 16 col

    ridge_regression = (
        linear_model
        .RidgeCV()
        .fit(X=A_train,y=train_subset.binary_key)
        )

    train_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_train),
        true_data = train_subset.binary_key
        )

    logger.info(f'\nalpha value:{np.round(ridge_regression.alpha_,3)}')
    logger.info(f'1,8 train Mean Square Error:{np.round(train_mean_square_error,4)}')
    

    indx = (test.y==1)|(test.y==8)

    test_subset = ImageData(
        category = '1,8 testing subset',
        x = test.x[indx],
        y = test.y[indx]
        )

    test_subset.binary_key = np.zeros(test_subset.y.shape)
    test_subset.binary_key[test_subset.y==1] = -1
    test_subset.binary_key[test_subset.y==8] =  1

    A_test = test_subset.x.dot(pca.components_[:16].transpose())  


    test_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_test),
        true_data = test_subset.binary_key
        )

    logger.info(f'1,8 test Mean Square Error:{np.round(test_mean_square_error,4)}')
    ###


    # part 4: 3 | 8
    indx = (train.y==3)|(train.y==8)

    train_subset = ImageData(
        category = '3,8 training subset',
        x = train.x[indx],
        y = train.y[indx]
        )

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==3] = -1
    train_subset.binary_key[train_subset.y==8] =  1

    A_train = train_subset.x.dot(pca.components_[:16].transpose())  

    ridge_regression = (
        linear_model
        .RidgeCV()
        .fit(X=A_train,y=train_subset.binary_key)
        )

    train_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_train),
        true_data = train_subset.binary_key
        )

    logger.info(f'\nalpha value:{np.round(ridge_regression.alpha_,3)}')
    logger.info(f'3,8 train Mean Square Error:{np.round(train_mean_square_error,4)}')
    

    indx = (test.y==3)|(test.y==8)

    test_subset = ImageData(
        category = '3,8 testing subset',
        x = test.x[indx],
        y = test.y[indx]
        )

    test_subset.binary_key = np.zeros(test_subset.y.shape)
    test_subset.binary_key[test_subset.y==3] = -1
    test_subset.binary_key[test_subset.y==8] =  1

    A_test = test_subset.x.dot(pca.components_[:16].transpose())  


    test_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_test),
        true_data = test_subset.binary_key
        )


    logger.info(f'3,8 test Mean Square Error:{np.round(test_mean_square_error,4)}')

    ### 2 | 7

    indx = (train.y==2)|(train.y==7)

    train_subset = ImageData(
        category = '2,7 training subset',
        x = train.x[indx],
        y = train.y[indx]
        )

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==2] = -1
    train_subset.binary_key[train_subset.y==7] =  1

    A_train = train_subset.x.dot(pca.components_[:16].transpose())  

    ridge_regression = (
        linear_model
        .RidgeCV()
        .fit(X=A_train,y=train_subset.binary_key)
        )

    train_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_train),
        true_data = train_subset.binary_key
        )

    logger.info(f'\nalpha value:{np.round(ridge_regression.alpha_,3)}')

    logger.info(f'2,7 train Mean Square Error:{np.round(train_mean_square_error,4)}')
    

    indx = (test.y==2)|(test.y==7)

    test_subset = ImageData(
        category = '2,7 testing subset',
        x = test.x[indx],
        y = test.y[indx]
        )

    test_subset.binary_key = np.zeros(test_subset.y.shape)

    test_subset.binary_key[test_subset.y==2] = -1
    test_subset.binary_key[test_subset.y==7] =  1

    A_test = test_subset.x.dot(pca.components_[:16].transpose())  


    test_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_test),
        true_data = test_subset.binary_key
        )

    logger.info(f'2,7 test Mean Square Error:{np.round(test_mean_square_error,4)}')
    ###







if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Config Path & Flags')

    parser.add_argument(
        '--data', metavar='data file souce path',
        type=str, nargs='?',
        help='input csv dir',
        default='../data'
        )

    parser.add_argument(
        '--plot', metavar='output plot directory path',
        type=str, nargs='?',
        help='output file dir',
        default='../plots'
        )

    parser.add_argument(
        '--log', metavar='log file destination path',
        type=str, nargs='?',
        help='log dir',
        default='../logs'
        )

    args = parser.parse_args()

    main(args=args)
