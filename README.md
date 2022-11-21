# ECS 171 Project: Ad Classification

## Preprocess

### Loading data

The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom unformatted document (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format. [The ad_data_parse notebook](./ad_data_parse.ipynb) achieves this.

## Imputing / Dropping

To impute and process our data, in the [data-imputing-minmax notebook](./data-imputing-minmax.ipynb), we evaluate the structure and annotation of our data.  We find that for image height, width, and aspect ratio, about 27% of the records have unknown values, so we impute them with the respective mean values of those features.  For the 'local' variable, less than 1% of the data has unknown values, so we simply drop these observations.

Additionally, the data is represented with fixed-digit string literals, with unknown values represented with a '?' character.  After imputing and dropping the unknown values, we must convert the data to numeric format.

### Encoding data

Once we convert the string values to numeric formatting, we can see that most of the features, such as url, are already one-hot encoded, so there's no additional encoding to be done.  Min-max scaling is unnecessary for binary data, but can be applied to our several non-binary encoded features, including height, width, and aspect ratio.  We therefore implement a min-max scaler to scale our data for more efficient computation and feature comparison in the future.

### Exploring and plotting

The next step is to get to know the shape of the data, the statistical attributes of each featrure, and its distribution. In order to do that, visual analysis can be conducted with the use of scatter plots and histograms in [this notebook](./data-exploration.ipynb). The following plots show the distribution of the continuous or non-binary variables: image width, height, and aspect ratio, stratified by the class (ad/non-ad) of the image.

![image](https://user-images.githubusercontent.com/37519138/202835315-090892b8-6d0a-45a2-ac63-aa27daae4087.png)

The datapoints appear clustered in the scatter plots, but from the histogram of the width and the ratio we can see a clear distinction in the distribution of ads vs non-ads.

### Correlation

We also measured the correlation of the continuous/non-binary variables to test for height, ratio, width, and class correlation, and interestingly, identified correlation between ad classification and image width.  This will likely be an important feature to include in our model building.

![download](https://user-images.githubusercontent.com/79494397/202986590-191cede4-5f83-49f5-890a-545f24b0c5db.png)


