# 171Project

## TODO: 

**How will you preprocess your data? You should explain this in your Readme.MD file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.**

## Preprocess

### Loading data

The used [dataset](https://archive.ics.uci.edu/ml/datasets/internet+advertisements) comes in a custom format (described in the [dataset documentation](./ad.DOCUMENTATION)), so the first step is to parse the dataset and convert it to the more familiar and Pandas-friendly .csv format. [This is the notebook](./ad_data_parse.ipynb) achieves this.

## Imputing / Dropping (???)

28% of the records of the dataset have at least a missing attribute (represented by the string '   ?'), so one possible choice is to drop them.

### Encoding data

The string values need now to be converted to iteger and floats. Most of the features of the dataset are alreasy one-hot encoded, so there's no additional encoding to be done.

# TODO: Normalization

### Exploring and plotting

The next step is to get to know the shape of the data, the statistical attributes of each featrure, and its distribution. In order to do that, visual analysis can be conducted with the use of scatter plots and histograms. The following plots show the distribution of the image width, height, and aspect ratio in relation to the class (ad/non-ad) of the image.

# TODO: ADD CORRELATION MATRIX??

![image](https://user-images.githubusercontent.com/37519138/202835315-090892b8-6d0a-45a2-ac63-aa27daae4087.png)

The datapoints appear clustered in the scatter plots, but from the histogram of the width and the ratio we can see a clear distinction in the distribution of ads vs non-ads.

