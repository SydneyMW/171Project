# ECS 171 Project: Final Writeup

## Introduction: Ad Classification Project

## Figures

### Pairplot

### Correlation Matrix

### Neural Net Training Curves

## Methods

### Data Pre-processing: Getting unstructured data into a structured format
The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom unformatted document (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format. The [ad_data_parse notebook](./1_ad_data_parse.ipynb) achieves this.

### Data Pre-Processing: Cleaning, imputing and scaling data
To impute and process our data, in the [preprocessing notebook](./3_preprocess_logreg_neuralnet.ipynb), we evaluate the structure and annotation of our data.  We find that for image height, width, and aspect ratio, about 27% of the records have unknown values, so we impute them with the respective mean values of those features.  For the 'local' variable, less than 1% of the data has unknown values, so we simply drop these observations.

Additionally, the data is represented with fixed-digit string literals, with unknown values represented with a '?' character.  After imputing and dropping the unknown values, we must convert the data to numeric format.

Once we convert the string values to numeric formatting, we can see that most of the features, such as url, are already one-hot encoded, so there's no additional encoding to be done.  Min-max scaling is unnecessary for binary data, but can be applied to our several non-binary encoded features, including height, width, and aspect ratio.  We therefore implement a min-max scaler to scale our data for more efficient computation and feature comparison in the future.

### Data Exploration: Visualizing data
...

### Model 1: Logistic Regression
...

### Model 2: Adversarial Neural Net Classifier
...

## Results
...

## Discussion
...

## Conclusion
...

## Collaboration

**Giulio:**

**Sarah:**

**Liudmila:**

**Henry:**

**Rongshan:**

**Sydney:** code to turn unformatted data into useable csv, code to perform scaling and imputing on data and generate pairplot, code for preliminary logistic regression and neural net models and evaluation, wrote readme “Model Fitting” section
