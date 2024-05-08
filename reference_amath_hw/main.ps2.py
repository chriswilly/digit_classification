"""
Michael Willy
AMATH Homework
January 2022
"""
import logging
import datetime
import pathlib
import numpy as np
np.set_printoptions(precision=2, threshold=7)

from sys import stdout
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import linear_model
from IPython import embed


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
class ImageData:
    category: str
    x: np.ndarray
    y: np.ndarray = None
    binary_key: np.ndarray = None


def mpl_params():
    """matplotlib parameters plot formatting
    """
    mpl.rcParams['figure.titlesize'] = 42
    mpl.rcParams['figure.titleweight'] = 'demibold'
    mpl.rcParams['axes.labelsize'] = 48
    mpl.rcParams['axes.titlesize'] = 40
    mpl.rcParams['xtick.labelsize'] = 48
    mpl.rcParams['ytick.labelsize'] = 48
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 40
    mpl.rcParams['lines.markeredgewidth'] = 3
    mpl.rcParams['legend.framealpha'] = 0.87
    mpl.rcParams['legend.fontsize'] = 36



def load_data(file:str,
              unsafe:bool=False
              )->np.ndarray:
    """ load numpy array files
    """
    file = pathlib.Path(file).resolve()

    if   isinstance(unsafe, bool) and not unsafe:
        allow_pickle = False
    elif isinstance(unsafe, bool) and unsafe:
        allow_pickle = True
    else:
        raise ValueError(f'unsafe:{type(unsafe)}:{unsafe}\n.')


    if file.exists():
        return np.load(file, allow_pickle=allow_pickle)
    else:
        raise ValueError(f'{file}\nNot located.')



def plot_line(data:np.ndarray,  # 1D in singular vals
              file_name:tuple = ('test','png'),
              labels:tuple = ('','')
              )->None:
    """
    """
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



def plot_digits(data:ImageData, 
                count:int, 
                title:str = None,
                file_name:tuple = (None,'png')
                ):
    """ Plot NxN digits
    """

    if not title:
        try:
            title = f'First {count} {data.category.title()}ing Features'
        except:
            title = f'First {count} Features'

    if not isinstance(file_name,tuple) or not file_name[0]: 
        file_name = (title,'png')

    output = pathlib.Path('plots').joinpath(f'{file_name[0]}.{file_name[-1]}')
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




def find_PCA_count(model:PCA, ratio:float)->int:
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

# def mean_square_error(predictor:np.ndarray,
#                       data:np.ndarray,
#                       output:np.ndarray)->float:
#     return 1/output.shape[0] * np.sum((data.dot(predictor) - output)**2)


def mean_square_error(prediction:np.ndarray,
                      true_data:np.ndarray)->float:
    return 1/true_data.shape[0] * np.sum((prediction - true_data)**2)



def main():
    """
        Instead of making a semantically pretty function chain above I'll just 
        pipe through the computation in main like a notebook, worst of both worlds :)
    """

    logger,_ = start_log()
    mpl_params()

    # return nested dict inside scalar np ndarray obj
    # Load and package data into dataclass
    test_dict  = load_data('data/MNIST_test_set.npy',unsafe=True)
    train_dict = load_data('data/MNIST_training_set.npy',unsafe=True)
    
    train = ImageData(category = 'train',
                      x = train_dict.item().get('features'),
                      y = train_dict.item().get('labels'))

    test  = ImageData(category = 'test',
                      x = test_dict.item().get('features'),
                      y = test_dict.item().get('labels'))

    logger.info(f'data loaded train: {train.x.shape}{train.y.shape}')
    logger.info(f'data loaded test:  {test.x.shape}{test.y.shape}')

    # init training model
    pca = PCA(n_components=train.x.shape[1])
    pca.fit(train.x)


    principal_components = ImageData(category='trained model',
                                     x = pca.components_)

    plot_digits(data=train,count=64)

    plot_digits(data=principal_components,
                count=16,
                file_name=('Principal Components','png'))



    # inspect singular values
    plot_line(data = np.log(pca.singular_values_),
              file_name=('Singular Values','png'),
              labels = ('index','log(Singular Values)')
              )


    plot_line(data = 100*pca.explained_variance_ratio_.cumsum(),
              file_name=('Cumulative Explained Variance Ratio','png'),
              labels = ('index','Cumulative Sum')
              )

    plot_line(data = 100*pca.explained_variance_ratio_.cumsum()[:16],
              file_name=('Cumulative Explained Variance Ratio Truncated','png'),
              labels = ('index','Cumulative Sum')
              )

    # estimate 
    thresholds = (0.6, 0.8, 0.9)
    # Explained variance approach - don't use lol
    [logger.info(f'{100*t}% index: {np.argmax(pca.explained_variance_ratio_.cumsum() >= t)}') for t in thresholds]

    # L2 / Frobenius norm of singular values
    pca_modes_threshold = {60:find_PCA_count(model=pca,ratio=0.6),
                           80:find_PCA_count(model=pca,ratio=0.8),
                           90:find_PCA_count(model=pca,ratio=0.9)
                           }
    # print('pca_modes_threshold:\n',pca_modes_threshold)
    [logger.info(f'{key}% of L2 norm with {val} PCA modes') for key,val in pca_modes_threshold.items()]
    

    # part 3: 1|8
    indx = (train.y==1)|(train.y==8)
    train_subset = ImageData(category = '1,8 training subset',
                             x = train.x[indx],
                             y = train.y[indx])

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==1] = -1
    train_subset.binary_key[train_subset.y==8] =  1

    # project X_1,8 onto first 16 PCA modes by dot product, 
    # PCA(n=16) is same as pca.components_[:16] so keep same model and just do the linalgebra if project -> dot project
    A_train = train_subset.x.dot(pca.components_[:16].transpose())  # 455,256 * 256,16  -> 455 row 16 col

    ridge_regression = (linear_model
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
    test_subset = ImageData(category = '1,8 testing subset',
                             x = test.x[indx],
                             y = test.y[indx])
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
    train_subset = ImageData(category = '3,8 training subset',
                             x = train.x[indx],
                             y = train.y[indx])

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==3] = -1
    train_subset.binary_key[train_subset.y==8] =  1

    A_train = train_subset.x.dot(pca.components_[:16].transpose())  

    ridge_regression = (linear_model
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
    test_subset = ImageData(category = '3,8 testing subset',
                             x = test.x[indx],
                             y = test.y[indx])

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
    train_subset = ImageData(category = '2,7 training subset',
                             x = train.x[indx],
                             y = train.y[indx])

    train_subset.binary_key = np.zeros(train_subset.y.shape)
    train_subset.binary_key[train_subset.y==2] = -1
    train_subset.binary_key[train_subset.y==7] =  1

    A_train = train_subset.x.dot(pca.components_[:16].transpose())  

    ridge_regression = (linear_model
                        .RidgeCV()
                        .fit(X=A_train,y=train_subset.binary_key)
                        )

    train_mean_square_error = mean_square_error(
                                prediction = ridge_regression.predict(A_train), 
                                true_data = train_subset.binary_key
                                )

    logger.info(f'\nalpha value:{np.round(ridge_regression.alpha_,3)}')
    logger.info(f'2,7 train Mean Square Error:{np.round(train_mean_square_error,4)}')
    

    indx = (test.y==3)|(test.y==8)
    test_subset = ImageData(category = '2,7 testing subset',
                             x = test.x[indx],
                             y = test.y[indx])
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



    # embed()




if __name__=='__main__':
    main()
