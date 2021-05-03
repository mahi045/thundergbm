#Data Preprocessing

### Data Crunching
The raw data is downloaded from https://transtats.bts.gov/ as zip file. 
The data_crunching.py script process the data and merge in geographical information of airports
and a binary variable indicating if the date is a public holiday.
The output of the script is a list of csv files, each containing the data of a month.
The dataset used in our experiment range from 01/2015-12/2020.

### Data Preprocessing
The data_preprocessing.py script process the categorical features of the data and convert 
the data to both the input format of ThunderGBM model and non-ThunderGBM model including 
XGBoost, LightGBM and CatBoost.

The script performs a two-step processing on the categorical features: 1. one hot encoding, 2. PCA.
To test out the performance of different PCA, we include both naive PCA and Incremental PCA
in the script and allow running on both GPU and CPU. We use a timer class to record the time elapse 
for each function.