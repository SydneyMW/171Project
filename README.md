# ECS 171 Project: Ad Classification

## Preprocess

### Loading data

The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom unformatted document (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format. [This notebook](./ad_data_parse.ipynb) achieves this.

## Imputing / Dropping

28% of the records of the dataset have at least one missing attribute, so one possible choice is to drop them.
Additionally, the data is represented with fixed-digit string literals, with unknown values represented with a '?' character.  After imputing or dropping the unknown values, we must convert the data to numeric format.

### Encoding data

Once we convert the string values to numeric formatting, we can see that most of the features, such as url, are already one-hot encoded, so there's no additional encoding to be done.  Min-max scaling is unnecessary for binary data so it is not needed for the vast majority of data.

### Exploring and plotting

The next step is to get to know the shape of the data, the statistical attributes of each featrure, and its distribution. In order to do that, visual analysis can be conducted with the use of scatter plots and histograms. The following plots show the distribution of the continuous or non-binary variables: image width, height, and aspect ratio, stratified by the class (ad/non-ad) of the image.

![image](https://user-images.githubusercontent.com/37519138/202835315-090892b8-6d0a-45a2-ac63-aa27daae4087.png)

The datapoints appear clustered in the scatter plots, but from the histogram of the width and the ratio we can see a clear distinction in the distribution of ads vs non-ads.

### Correlation

We also measured the correlation of the continuous/non-binary variables to test for height, ratio, width, and class correlation, and interestingly, identified correlation between ad classification and image width.  This will likely be an important feature to include in our model building.



