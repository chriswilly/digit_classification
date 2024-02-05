# ps2 notes


1. Use PCA to investigate the dimensionality of $X_{\text {train }}$ and plot the first 16 PCA modes as $16 \times 16$ images.
-ok

2. How many PCA modes do you need to keep in order to approximate $X$ train up to $60 \%, 80 \%$ and $90 \%$ in the Frobenius norm? Recall the identity

60% of L2 norm with 3 PCA modes
80% of L2 norm with 7 PCA modes
90% of L2 norm with 14 PCA modes




3. Train a classifier to distinguish the digits 1 and 8 via the following steps:

- First, you need to write a function that extracts the features and labels of the digits 1 and 8 from the training data set. Let us call these $X_{(1,8)}$ and $Y_{(1,8)}$.
--ok



- Then project $X_{(1,8)}$ on the first 16 PCA modes of $X_{\text {train }}$ computed in step 1 , this should give you a matrix $A_{\text {train }}$ which has 16 columns corresponding to the PCA coefficients of each feature and 455 rows corresponding to the total number of 1 's and 8 's in the training set. .
--ok



- Assign label $-1$ to the images of the digit 1 and label $+1$ to the images of the digit 8 . This should result in a vector $b_{\text {train }} \in\{-1,+1\}^{455}$.
--ok


- Use Ridge regression or least squares to train a predictor for the vector $b_{\text {train }}$ by linearly combining the columns of $A_{\text {train. }}$.
--ok


- Report the training mean squared error (MSE) of your classifier

- Report the testing MSE of your classifier
where you need to construct analogously the matrix $A_{\text {test }}$ and $b_{\text {test }}$ corresponding to the digits 1 and 8 from the test data set.

4. Use your code from step 3 to train classifiers for the pairs of digits $(3,8)$ and $(2,7)$ and report the training and test MSE's. Can you explain the performance variations?










1. sklearn has functionalities for PCA and Ridge regression with crossvalidation. Using these can make your life much easier.

2. Don’t forget to center 𝑋train before computing the PCA modes if you plan to use SVD. If you are using sklearn’s PCA function then you don’t need to worry about this as it centers the data by default.
--ok




3. Make note that in task 3 you will be projecting your sub-training and sub-test sets corresponding to specific digits on the PCA modes that were computed on the entire training set.

4. In step 3 we assign labels -1 and +1 to images of the digits 1 and 8. This is done to make the output of your classifier normalized and is common practice in binary classification where we wish to distinguish only two classes in a data set.



--First perform PCA on all of the training images (all digits, not just 1s and 8s) and take the first 16 modes. Other choices could be made when building a classfier, but for this homework task you are told to use the first 16 components from performing PCA on all digits so that we can easily check that you got the expected results.

--Next pull the relevant training images (1s and 8s) in order to build the training set for your classfier. Also pull the relevant training labels and convert to -1 and +1 instead of 1s and 8s.

--Project the training set onto the 16 PCA components. Now your training images are each represented by a point in the 16-dimensional component space. 

--Train your classifier to associate points in 16 dimensional space with either the label +1 or =1 by fitting it to the training data and training labels.



__________

___________________

Hints for Homework 2

1. Task 1:

○ Use PCA to investigate the dimensionality of 𝑋_train
■ Make a plot of the singular value spectrum. You do not need to assign a specific dimension to the data, but generally discuss what you notice from this plot.
--ok

○ Plot the first 16 PCA modes as 16 × 16 images
--ok

2. Task 2: How many PCA modes do you need to keep in order to approximate 𝑋_train up
to 60%, 80% and 90% in the Frobenius norm? Do you need the entire 16 × 16 image for each data point?

○ Be careful to use the frobenius norm and not the square of frobenius norm when calculating your values.
--ok?


○ If you normalize the data so that every pixel has variance of 1 you will find slightly higher number of modes needed. Because all of the pixels are on the same order, you can get reasonable results for the following parts without standardizing. It is also fine to standardize things.


○ You could look at a few example digits and see how recognizable they are at different low rank reconstructions if you want to.
--


3. Task
○ For debugging purposes, check your dimensions at each step!

○ Note that the PCA modes should be generated from the full X_train data set, not
just the 1s and 8s.
--ok

○ You are welcome to use all the built in functionality from sklearn PCA
implementation including transform for building A_train and A_test.


○ Remember that predicting with your classifier is a little different that taking A*beta
because when you predict you round up to +1 or down to -1 depending on the
sign. Use the MSE formula as written or equivalent built in functions.

○ It is advised to use cross validation! In python use RidgeCV as shown in lecture.

○ Making plots of the training and test error for different values of alpha could be
nice if you have room. You could also make a table of your final MSEs if you do not. Make sure to report parameter values used in your training in your algorithm implementation and development section.

3: This task is outlined for you on the HW.

 4. Task
MSEs and speculate as to why they differ. If you want to investigate further you could think about the following.

4: The bare minimum here is to repeat task 3 with the other pairs of digits, compare

○ When comparing results for your different classifiers, it could be interesting to visualize where the 1s and 8s live in the PCA space. You could plot their position using the coefficients for modes 1, 2 and 3 and see whether the clouds are close together or far apart. Same thing for 3s and 8s vs 2s and 7s.

○ You could make confusion matrices. See sklearn.metrics.confusion_matrix

○ You could visualize the weights of beta after you fit the ridge classifier. Which
modes are most important in each case?

○ Did you have a balanced number of training digits in each case?
