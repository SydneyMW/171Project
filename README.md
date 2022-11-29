# ECS 171 Project: Ad Classification

## Milestone 2 - Preprocess

### Loading Data

The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom unformatted document (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format. The [ad_data_parse notebook](./1_ad_data_parse.ipynb) achieves this.

## Imputing / Dropping

To impute and process our data, in the [preprocessing notebook](./3_preprocess_logreg_neuralnet.ipynb), we evaluate the structure and annotation of our data.  We find that for image height, width, and aspect ratio, about 27% of the records have unknown values, so we impute them with the respective mean values of those features.  For the 'local' variable, less than 1% of the data has unknown values, so we simply drop these observations.

Additionally, the data is represented with fixed-digit string literals, with unknown values represented with a '?' character.  After imputing and dropping the unknown values, we must convert the data to numeric format.

### Encoding and Scaling Data

Once we convert the string values to numeric formatting, we can see that most of the features, such as url, are already one-hot encoded, so there's no additional encoding to be done.  Min-max scaling is unnecessary for binary data, but can be applied to our several non-binary encoded features, including height, width, and aspect ratio.  We therefore implement a min-max scaler to scale our data for more efficient computation and feature comparison in the future.

### Exploring and Plotting

The next step is to get to know the shape of the data, the statistical attributes of each featrure, and its distribution. In order to do that, visual analysis can be conducted with the use of scatter plots and histograms in the [data-exploration notebook](./2_data_exploration.ipynb). The following plots show the distribution of the continuous or non-binary variables: image width, height, and aspect ratio, stratified by the class (ad/non-ad) of the image.

![image](https://user-images.githubusercontent.com/37519138/202835315-090892b8-6d0a-45a2-ac63-aa27daae4087.png)

The datapoints appear clustered in the scatter plots, but from the histogram of the width and the ratio we can see a clear distinction in the distribution of ads vs non-ads.

### Correlation

We also measured the correlation of the continuous/non-binary variables to test for height, ratio, width, and class correlation, and interestingly, identified correlation between ad classification and image width.  This will likely be an important feature to include in our model building.

![image](https://user-images.githubusercontent.com/37519138/204183883-2f1ec76b-3907-4616-9d80-2567d45840af.png)

## Milestone 3 - Model construction

### Logistic Regression Model

We decided to implement logistic regression first, due to the binary classification structure of our problem. The model is relatively simple, which will enable it to handle the many features in our dataset and use straightforward iteration to update weights for ad/non-ad classification predictions.

The logistic regression model is developed in the [modeling notebook](./3_preprocess_logreg_neuralnet.ipynb). We fit the model using the min-max scaled X_train and y_train. With our training data, we were able to achieve an accuracy of 0.99, and our testing data gave us an accuracy of 0.98. This was highly successful, with similar training and testing error, indicating both the adequacy of our model for the problem and the quality of the data for making such a classification.  Out of curiosity, we decided to see how a more complex model would compare.

### Simple Neural Net Model

The second model we created, also in the [modeling notebook](./3_preprocess_logreg_neuralnet.ipynb), is a neural net model with 1 input layer, 2 hidden layers, and 1 output layer. The input layer has 16 nodes and uses relu activation function. The 2 hidden layers also use relu activation functions, but the first uses 8 nodes and the second hidden layer uses 6 nodes. Finally, the output layer uses a sigmoid activation function. With our neural network model, we were able to achieve a testing accuracy of 0.99.  The high training and testing performance of the neural net despite its unnecessary complexity indicates that we may attribute both of our model's success to the quality of the data fed to each, since both simple and complex models are able to perform well with little optimization.
