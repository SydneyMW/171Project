# ECS 171 Ad Classification Project: Final Writeup

## Introduction
...

## Figures

### Figure 1 &mdash; Pairplot
...
![image](https://user-images.githubusercontent.com/37519138/202835315-090892b8-6d0a-45a2-ac63-aa27daae4087.png)

### Figure 2 &mdash; Correlation Matrix
...
![image](https://user-images.githubusercontent.com/37519138/204183883-2f1ec76b-3907-4616-9d80-2567d45840af.png)

### Figure 3 &mdash; Neural Net Training Curves
...
![image](https://user-images.githubusercontent.com/37519138/205465185-981df9ed-3c46-440e-bb3d-24583296ba08.png)
![image](https://user-images.githubusercontent.com/37519138/205465190-b6bce9be-270f-49a1-9f5a-422972cdd67c.png)


## Methods

### 1. Data Pre-processing &mdash; Getting unstructured data into a structured format
The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom unformatted document (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format.  We use the features described in the [names file](./dataset/ad.names) to convert the [data](./dataset/ad.data) to a pandas dataframe, which we save as our [data csv file](./dataset/data.csv). 
The original data csv contains the observations but has no column names and looks like this:
|    |    0 |    1 |      2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |
|---:|-----:|-----:|-------:|----:|----:|----:|----:|----:|----:|----:|
|  0 |  125 |  125 | 1      |   1 |   0 |   0 |   0 |   0 |   0 |   0 |
|  1 |   57 |  468 | 8.2105 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |
|  2 |   33 |  230 | 6.9696 |   1 |   0 |   0 |   0 |   0 |   0 |   0 |
|  3 |   60 |  468 | 7.8    |   1 |   0 |   0 |   0 |   0 |   0 |   0 |
|  4 |   60 |  468 | 7.8    |   1 |   0 |   0 |   0 |   0 |   0 |   0 |

Whereas the names file contains unstructured feature names in this format:
|    | Original                                                  |
|---:|:----------------------------------------------------------|
|  0 | | "w:\c4.5\alladA" names file -- automatically generated! |
|  1 |                                                           |
|  2 | ad, nonad | classes.                                      |
|  3 |                                                           |
|  4 | height: continuous.                                       |
|  5 | width: continuous.                                        |
|  6 | aratio: continuous.                                       |
|  7 | local: 0,1.                                               |
|  8 | | 457 features from url terms                             |
|  9 | url\*images+buttons: 0,1.                                  |
| 10 | url\*likesbooks.com: 0,1.                                  |
| 11 | url\*www.slake.com: 0,1.                                   |

The names are in order, and must be parsed and assigned to the dataframe to create the final csv of our features and observations.
This data processing and csv creation is accomplished in the [ad_data_parse notebook](./1_ad_data_parse.ipynb).

### 2. Data Pre-Processing &mdash; Cleaning, imputing and scaling data
Once we have formatted data, we want analyze its features, check for null or unknown values, and perform dropping/imputing/scaling as needed.  We find that the dataset has over 1500 features and over 3000 observations, with all but four features (height, width, aratio, and local) being binary-encoded.  Those that are not binary-encoded are represented with fixed-digit string literals due to the odd unstructured format of the original document, with unknown values represented by a '?' character.  

We find that for image height, width, and aspect ratio, about 27% of the records have unknown values, so we impute them with the respective mean values of those features.  For the 'local' variable, less than 1% of the data has unknown values, so we simply drop these observations.  After imputing and dropping the unknown values, we must convert the data to numeric format.

Once we convert the string values to numeric formatting, we can see that most of the features, such as url, are already one-hot encoded, so there's no additional encoding to be done.  Min-max scaling is unnecessary for binary-encoded data, but can be applied to our several non-binary-encoded features, including height, width, and aspect ratio.  We therefore implement a min-max scaler to scale our data for more efficient computation and feature comparison in the future.  We choose min-max scaling due to its simplicity and the lack of normal distribution among all continuous variables, shown by our pairplot in the data exploration section (described next).

This processing, imputing, and scaling is performed in the [preprocessing notebook](./3_preprocess_logreg_neuralnet.ipynb) prior to model development.

### 3. Data Exploration &mdash; Visualizing data
The next step is to get to know the shape of the data, the statistical attributes of each featrure, and its distribution. In order to do that, visual analysis can be conducted with the use of a pairplot and correlation matrix in the [data_exploration notebook](./2_data_exploration.ipynb). 

The pairplot (Figure 1) shows the distribution of the height, width, and aratio variables, grouped by ad/non-ad classification.  The datapoints appear highly overlapping in the scatter plots, but from the histogram of the width and the aratio, we can see a clear distinction in the distribution of ads vs non-ads, with ads showing bimodal distribution for width and aratio, and non-ads showing roughly normal distribution.

The correlation matrix (Figure 2) shows the correlation coefficients between these same features.  Notable, there is a high correlation between ad classification and image width, so width will certainly be an important feature to include in our model building.

### 4. Model 1 &mdash; Logistic Regression
...

### 5. Model 2 &mdash; Adversarial Neural Net Classifier
...

### 6. Model 3 &mdash; Support Vector Machine
...

## Results
...

## Discussion
...

## Conclusion
...

## Collaboration

**Giulio:** code for SVM classification, improved Neural Network training curve plots, code for exploratory histogram plot, wrote 1st version of the README

**Sarah:**

**Liudmila:**

**Henry:**

**Rongshan:**

**Sydney:** code to turn unformatted data into useable csv, code to perform scaling and imputing on data and generate pairplot, code for preliminary logistic regression and neural net models and evaluation, wrote readme “Model Fitting” section
