

## PCA classifier

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

    logger.info(f'{a},{b} train Mean Square Error:{np.round(100*train_mean_square_error,4)}%')


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

    logger.info(f'{a},{b} test Mean Square Error:{np.round(100*test_mean_square_error,4)}%')




## histograms:

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





## Graph Laplacian

    ## Spectral Clustering Graph Laplacian training model
    graph_laplacian = SpectralClustering(
        n_clusters = 10, ## <--
        n_components = 10, ## TODO: tune for > number of classes and
        affinity = 'rbf',
        # gamma = 1.0,
        n_jobs = -1,
        assign_labels = 'cluster_qr'
        )

    graph_laplacian.fit(train.x)

    logger.info(f'{np.unique(train.y)=}')
    logger.info(f'{np.unique(graph_laplacian.labels_)=}')

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


