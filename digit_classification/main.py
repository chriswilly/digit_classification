"""
Michael Willy
AMATH582 Homework 2 + 4 combined digit classifier
January 29, 2022 update February 2, 2024
"""
import argparse
import logging
from datetime import datetime
import pathlib
import numpy as np

from rich import print
from sys import stdout
from dataclasses import (
    dataclass,
    field
    )

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from sklearn.decomposition import PCA
from sklearn import linear_model

from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from sklearn.base import clone

# custom but write this out
from graph_laplacian import GraphLaplacian

import IPython
np.set_printoptions(precision=2, threshold=27)


## Container class

@dataclass
class ImageData:
    category: str
    x: np.ndarray
    y: np.ndarray = None
    binary_key: np.ndarray = None


## Utils

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
    handler = logging.StreamHandler(stdout)
    logging_formatter = logging.Formatter(r'%(asctime)s:%(name)s:%(message)s')
    handler.setFormatter(logging_formatter)
    logger.addHandler(handler)

    return logger, log_file


def mpl_params()->None:
    """ matplotlib parameters plot formatting
    """
    mpl.rcParams['figure.titlesize'] = 32
    mpl.rcParams['figure.titleweight'] = 'demibold'
    mpl.rcParams['axes.labelsize'] = 32
    mpl.rcParams['axes.titlesize'] = 24
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


## Plots

def plot_line(
    data:np.ndarray,  # 1D in singular vals
    file_name:str,
    ext:str='.png',
    labels:tuple = ('','')
    )->None:
    """
    """

    output = (
        pathlib.Path('../plots')
        .joinpath(file_name).with_suffix(ext)
        )

    output.resolve().parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(13.5,12))
    ax = fig.add_subplot(111)

    if len(data.shape)>1:
        ax.plot(data[...,0],data[...,1],'-k',label=f'')

    else:
        ax.plot(data,'-k',label=f'')


    plt.axis('tight')
    plt.title(file_name)
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
            title = f'First {count} {data.category.title()} Features'
        except:
            title = f'First {count} Features'

    output = (
        pathlib.Path('../plots')
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
        ax[indx].imshow(
            (data.x[k,:].reshape((pixel_length, pixel_length))),
            cmap='Greys'
            )
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
        pathlib.Path('../plots')
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
        pathlib.Path('../plots')
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
            pathlib.Path('../plots')
            .joinpath(f'{file_name}/{file_name}_{indx}')
            .with_suffix(ext)
            )

        plt.grid(visible=True, which='major', axis='both')
        fig.savefig(output, bbox_inches='tight')
        plt.close('all')



def histogram_plot(
    data:ImageData,
    file_name:str,
    mode:str,
    labels:tuple = ('',''),
    ext:str='.png',
    PAPER_SIZE:tuple=(11,6)
    )->None:
    """
    """
    output = (
        pathlib.Path('../plots')
        .joinpath(file_name)
        )

    output.resolve().mkdir(parents=True, exist_ok=True)

    digits = np.unique(data.x)

    for number in digits:

        indx = (data.x == number)&(data.binary_key == True)
        indy = (data.y == number)&(data.binary_key == True)

        # todo find a better enum or switch vs this str key

        # cross compare
        if mode == 'feature':
            frequency_data = data.y[indx]

        elif mode == 'prediction':
            frequency_data = data.x[indy]

        else:
            raise Exception(f'{mode} key is not in enum "feature" or "prediction"')


        fig = plt.figure(figsize=PAPER_SIZE)
        ax = fig.add_subplot(111)

        count,bins = np.histogram(frequency_data, bins = np.arange(11))
        # func = func/func.sum() # pdf
        # print(number)
        # print(bins)
        # print(count)

        bins = bins[:-1]

        ax.bar(
            bins,count,
            width = 0.85,
            label = list(map(str,digits)),
            alpha = 0.67
            )

        padding = 0.5
        limits = [
            np.floor(digits.min()),
            np.ceil( digits.max()),
            0,
            count.max()+padding
            ]

        plt.axis(limits)

        plt.title(f'{file_name} {number}')

        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

        # ax.legend(loc=0)

        ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=(limits[3]-limits[2])/4))

        output = (
            pathlib.Path('../plots')
            .joinpath(f'{file_name}/{file_name}_{number}')
            .with_suffix(ext)
            )

        plt.grid(visible=True, which='major', axis='both')
        fig.savefig(output, bbox_inches='tight')
        plt.close('all')



def find_pca_count(
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

    raise Exception(
        f"""exceeded {model.singular_values_.shape[0]} modes,
         check your ratio value {ratio} as the threshold should be met"""
         )


def mean_square_error(
    prediction:np.ndarray,
    true_data:np.ndarray
    )->float:
    """
    """
    return 1/true_data.shape[0] * np.sum((prediction - true_data)**2)


def run_binray_pca_classifier(
    digits:tuple,
    train:ImageData,
    test:ImageData,
    pca:PCA,
    logger:logging.Logger,
    components:int=16,
    )->...:
    """
    """
    assert len(digits) == 2
    a,b = digits

    indx = (train.y==a)|(train.y==b)

    train_subset = ImageData(
        category = f'{a},{b} training subset',
        x = train.x[indx],
        y = train.y[indx]
        )

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==a] = -1
    train_subset.binary_key[train_subset.y==b] =  1

    A_train = train_subset.x.dot(pca.components_[:components].transpose())

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

    logger.info(f'{a},{b} train Mean Square Error:{np.round(train_mean_square_error,4)}')


    indx = (test.y==a)|(test.y==b)

    test_subset = ImageData(
        category = f'{a},{b} testing subset',
        x = test.x[indx],
        y = test.y[indx]
        )

    test_subset.binary_key = np.zeros(test_subset.y.shape)

    test_subset.binary_key[test_subset.y==a] = -1
    test_subset.binary_key[test_subset.y==b] =  1

    A_test = test_subset.x.dot(pca.components_[:components].transpose())


    test_mean_square_error = mean_square_error(
        prediction = ridge_regression.predict(A_test),
        true_data = test_subset.binary_key
        )

    logger.info(f'{a},{b} test Mean Square Error:{np.round(test_mean_square_error,4)}')




def main(args:argparse.Namespace)->None:
    """
    """

    logger,log_file = start_log(
        name=f'{pathlib.Path(__file__).stem}',
        destination=args.log
        )

    mpl_params()

    ## Load Data & make dataclass
    test_labels = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_test_labels.csv'),
        dtype=float,
        delimiter=','
        ).astype(int)

    tain_labels = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_training_labels.csv'),
        dtype=float,
        delimiter=','
        ).astype(int)

    test_features = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_test_features.csv'),
        dtype=float,
        delimiter=','
        )

    train_features = np.loadtxt(
        fname=pathlib.Path(args.data).joinpath('MNIST_training_features.csv'),
        dtype=float,
        delimiter=','
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


    plot_digits(
        data=train,
        count=64,
        file_name='Training Features'
        )

    ## Principal Component Analysis training model
    pca = PCA(n_components=train.x.shape[1])
    pca.fit(train.x)

    principal_components = ImageData(
        category='trained model',
        x = pca.components_
        )

    plot_digits(
        data=principal_components,
        count=16,
        file_name='Principal Components'
        )

    # L2 / Frobenius norm of singular values
    pca_modes_threshold = {
        60:find_pca_count(model=pca,ratio=0.6),
        80:find_pca_count(model=pca,ratio=0.8),
        90:find_pca_count(model=pca,ratio=0.9)
        }

    # print('pca_modes_threshold:\n',pca_modes_threshold)
    [logger.info(f'{key}% of L2 norm with {val} PCA modes') for key,val in pca_modes_threshold.items()]

    ## Binary classifier
    for digit_pair in ((1,8),(3,8),(2,7),(4,7),):
        run_binray_pca_classifier(
            digits = digit_pair,
            train = train,
            test = test,
            pca = pca,
            logger = logger,
            components=24,
            )


    ## Spectral Clustering Graph Laplacian training model
    graph_laplacian = SpectralClustering(
        n_clusters = 10,
        n_components = 10, ## TODO: tune for > number of classes and
        affinity = 'rbf',
        # gamma = 1.0,
        n_jobs = -1,
        assign_labels = 'cluster_qr'
        )

    graph_laplacian.fit(train.x)

    # reassign true labels
    ## TODO: devise way to have distinct labels pointing to similar true label, e.g. {1:1,22:1,2:2,29:2, ...}

    for label in np.unique(train.y):
        indx = train.y==label
        graph_laplacian.labels_[indx] = label


    graph_laplacian_classifier = SpectralEmbedding(
        n_components = graph_laplacian.n_components,
        affinity = graph_laplacian.affinity,
        # gamma = None, ## TODO: tune gamma for accuracy
        n_jobs = -1,
        )

    graph_laplacian_classifier.fit(train.x)


    print(graph_laplacian.labels_[:100])
    print(train.y[:100])

    # IPython.embed()

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



    train.binary_key = np.zeros([train.y.shape[0],10],dtype=int)

    # lookup how to ravel & put vs this row by row loop
    for indx in np.arange(train.y.size):
        train.binary_key[indx,train.y[indx]-1] = 1


    #sample_size = -1
    #eigen_depth = -1
    #[:sample_size,:eigen_depth]

    # [0 1]*2 - 1 --> +1 / -1 one-hot instead of 1 / 0, this keeps symmetric about origin

    trained_predictor_matrix = np.linalg.lstsq(
        a=graph_laplacian_classifier.embedding_,
        b=(train.binary_key*2 - 1),
        rcond=None
        )[0]

    regression_predictor = graph_laplacian_classifier.embedding_.dot(trained_predictor_matrix)

    regression_classifier = np.zeros(regression_predictor.shape)

    for indx,row in enumerate(regression_predictor):
        indy = np.argmax(row)
        regression_classifier[indx,indy] = 1

    reconstructed_train_labels = (1 + regression_classifier.dot(np.arange(0,10))).astype(int)

    ## Evaluate test data all obj with test_ prefix

    test_graph_laplacian_classifier = clone(graph_laplacian_classifier)

    test_graph_laplacian_classifier.fit(test.x)

    test_regression_predictor = test_graph_laplacian_classifier.embedding_.dot(trained_predictor_matrix)

    test_regression_classifier = np.zeros(test_regression_predictor.shape)

    for indx,row in enumerate(test_regression_predictor):
        indy = np.argmax(row)
        test_regression_classifier[indx,indy] = 1

    reconstructed_test_labels = (1 + test_regression_classifier.dot(np.arange(0,10))).astype(int)

    ## evaluate error / accuracy rate
    graph_laplacian_error = {
        'train':[0,1],
        'test': [0,1]
        }

    mislabeled_train = reconstructed_train_labels!=train.y

    mislabeled_test = reconstructed_test_labels!=test.y

    classification_train_error = ImageData(
        category = 'train label error',
        x = train.y,
        y = reconstructed_train_labels,
        binary_key = mislabeled_train
        )

    # classification_test_error = ImageData(
    #     category = 'test label error',
    #     x = test.y,
    #     y = reconstructed_test_labels,
    #     binary_key = mislabeled_test
    #     )


    graph_laplacian_error['train'][0] = np.sum(mislabeled_train)

    graph_laplacian_error['train'][1] = graph_laplacian_error['train'][0] / train.y.size

    graph_laplacian_error['test'][0] = np.sum(mislabeled_test)

    graph_laplacian_error['test'][1] = graph_laplacian_error['test'][0] / test.y.size

    logger.info(f"train:{100*graph_laplacian_error['train'][1]}% across {train.y.size} samples")

    logger.info(f"test:{100*graph_laplacian_error['test'][1]}% across {test.y.size} samples")


    # IPython.embed()

    histogram_plot(
        data = classification_train_error,
        file_name = 'Training Classification Error on Features',
        mode = 'feature',
        labels = ('Feature Digits','Mislabeled Count')
        )


    histogram_plot(
        data = classification_train_error,
        file_name = 'Training Classification Error on Predictions',
        mode = 'prediction',
        labels = ('Prediction Digits','Mislabeled Count')
        )

    histogram_plot(
        data = classification_test_error,
        file_name = 'Test Classification Error on Features',
        mode = 'feature',
        labels = ('Feature Digits','Mislabeled Count')
        )


    histogram_plot(
        data = classification_test_error,
        file_name = 'Test Classification Error on Predictions',
        mode = 'prediction',
        labels = ('Prediction Digits','Mislabeled Count')
        )

    ### Exclude similar digits
    indx = (train.y!=3)&(train.y!=8)&(train.y!=9)
    indy = (test.y!=3)&(test.y!=8)&(test.y!=9)

    graph_laplacian_error_exclusion = {
        'train':[0,1],
        'test': [0,1]
        }

    graph_laplacian_error_exclusion['train'][0] = np.sum(mislabeled_train[indx])

    graph_laplacian_error_exclusion['train'][1] = graph_laplacian_error_exclusion['train'][0] / train.y[indx].size

    graph_laplacian_error_exclusion['test'][0] = np.sum(mislabeled_test[indy])

    graph_laplacian_error_exclusion['test'][1] = graph_laplacian_error_exclusion['test'][0] / test.y[indy].size

    logger.info(f"train:{100*graph_laplacian_error_exclusion['train'][1]}% across {train.y[indx].size} samples")

    logger.info(f"test:{100*graph_laplacian_error_exclusion['test'][1]}% across {test.y[indy].size} samples")



    ## Nix this hand made graph laplacian, but was used for some plots

    # scale_factor = 0.105
    # laplacian = GraphLaplacian(
    #     data=train.x,
    #     distance_ratio=scale_factor
    #     )
    # indx = (laplacian.eigval<=5e-20)
    # print(f'{np.sum(indx)} distict groups given {scale_factor} relative distance')

    # eigval_plot(
    #     data = np.log(laplacian.eigval[1:]),
    #     file_name = 'Graph Laplacian Eigenvalues',
    #     labels = ('$Index_j$',r'$log(\lambda_j)$'),
    #     marker ='b.',
    #     arbitrary_sigma = laplacian.length_scale
    #     )

    # eigval_plot(
    #     data = np.log(laplacian.eigval_norm[1:]),
    #     file_name = 'Graph Laplacian Eigenvalues Normalized',
    #     labels= ('$Index_j$',r'$log(\lambda_j)$'),
    #     marker='b.',
    #     arbitrary_sigma = laplacian.length_scale
    #     )

    # eigvec_plot(
    #     graph=laplacian,
    #     file_name = 'Graph Laplacian Eigenvectors',
    #     depth=4,
    #     )

    # eigvec_3d_plot(
    #     graph=laplacian,
    #     file_name = 'Graph Laplacian Eigenvectors',
    #     depth=2,
    #     )

    # eigval_plot(
    #     data = 1000*np.diff(laplacian.eigval)[:8],
    #     file_name = 'Change in Eigenvalues',
    #     labels = ('$Index_m$',r'$\Delta\lambda$'),
    #     marker='k-',
    #     arbitrary_sigma = laplacian.length_scale
    #     )


    ## PCA % ANOVA
    ## Explained variance approach - don't use lol
    # thresholds = (0.6, 0.8, 0.9)
    # [logger.info(f'{100*t}% index: {np.argmax(pca.explained_variance_ratio_.cumsum() >= t)}') for t in thresholds]




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Config Path & Flags')

    parser.add_argument(
        '--data', metavar='data file souce path',
        type=str, nargs='?',
        help='input csv dir',
        default='../data'
        )

    parser.add_argument(
        '--log', metavar='log file destination path',
        type=str, nargs='?',
        help='log dir',
        default='../logs'
        )

    # parser.add_argument(
    #     '--plot', metavar='output plot directory path',
    #     type=str, nargs='?',
    #     help='output file dir',
    #     default='../plots'
    #     )

    args = parser.parse_args()

    main(args=args)
