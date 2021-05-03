from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
import cuml
from sklearn.pipeline import Pipeline

import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm

from timer import Timer

from collections import Counter
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Preprocessor(object):
    def __init__(self, n_components, batch_size, use_gpu, incremental=True, output_dir="output/preprocessing"):
        self.n_components = n_components
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.incremental = incremental
        self.pca = None
        self.enc = None

        self.pca_path = f'{output_dir}/pca_gpu_{self.use_gpu}_incremental_{self.incremental}_bs_{self.batch_size}.pickle'
        self.enc_path = f'{output_dir}/one_hot_encoder.pickle'

    def train_one_hot_encoding(self, df):
        self.enc = OneHotEncoder(handle_unknown='ignore')

        self.enc.fit(df)
        with open(self.enc_path, 'wb') as handle:
            pickle.dump(self.enc, handle)

    def train_pca(self, encoded_arr):
        try:
            if self.incremental:
                print(f"incremental PCA with batch size {self.batch_size}")
                if self.use_gpu:
                    self.pca = cuml.decomposition.IncrementalPCA(n_components=self.n_components,
                                                                 batch_size=self.batch_size)
                    for idx in tqdm(range(0, encoded_arr.shape[0], self.batch_size)):
                        self.pca.fit(encoded_arr[idx:min(idx + self.batch_size, encoded_arr.shape[0])])
                else:
                    self.pca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
                    self.pca.fit(encoded_arr[:self.batch_size])
            else:
                if self.use_gpu:
                    self.pca = cuml.decomposition.PCA(n_components=self.n_components)
                else:
                    self.pca = PCA(n_components=self.n_components)
                self.pca.fit(encoded_arr.toarray()[:self.batch_size])
        except Exception as e:
            print(
                f'Fail to run PCA with batch size: {self.batch_size}, use_gpu: {self.use_gpu}, incremental: {self.incremental}')
            raise e
        print(f'save pca to {self.pca_path}')
        with open(self.pca_path, 'wb') as handle:
            pickle.dump(self.pca, handle)

    def one_hot_encoding(self, df):
        if self.enc is None:
            with open(self.enc_path, 'rb') as handle:
                self.enc = pickle.load(handle)
        encoded_df = self.enc.transform(df)
        return encoded_df

    def run_pca(self, arr):
        if self.pca is None:
            with open(self.pca_path, 'rb') as handle:
                self.pca = pickle.load(handle)
            print(f'batch size: {self.pca.batch_size}')
        arr_list = []
        for idx in tqdm(range(0, arr.shape[0], self.batch_size)):
            df_arr_i = self.pca.transform(arr[idx:min(idx + self.batch_size, arr.shape[0])])
            arr_list.append(df_arr_i)
        dr_arr = np.concatenate(arr_list)
        return dr_arr


def load_flight_data(train_path="", test_path="", is_save_data=True):
    if is_save_data:
        assert train_path and test_path

    # create complete dataset
    df = pd.DataFrame()
    for fpath in fpath_list:
        temp_df = pd.read_csv(fpath)[columns]
        df = df.append(temp_df)

    # extract hour
    df['CRSDepTime'] = (df['CRSDepTime'] / 100).astype(int)
    df['CRSArrTime'] = (df['CRSArrTime'] / 100).astype(int)

    # set datatype for categorical columns
    for col, col_dtype in cat_cols_mapper.items():
        df[col] = df[col].astype(col_dtype)

    np.random.seed(27)
    # split data to train/test
    msk = np.random.rand(len(df)) < 0.7
    train = df[msk]
    test = df[~msk]
    sample = train[:1000]
    print(f"Train: {Counter(train['is_delayed'])}, test: {Counter(test['is_delayed'])}")

    if is_save_data:
        train.to_parquet('data/raw/raw_train.parquet.gzip',
                         index=False,
                         compression='gzip')

        test.to_parquet('data/raw/raw_test.parquet.gzip',
                        index=False,
                        compression='gzip')

        sample.to_parquet('data/raw/raw_sample.parquet.gzip',
                          index=False,
                          compression='gzip')
    return train, test


def to_tgbm_input(num_df, cat_arr, writePath):
    batch_size = 100000
    for i in tqdm(range(0, len(num_df), batch_size)):
        end_i = min(i + batch_size, len(num_df))
        arr = np.concatenate((num_df[i:end_i].values, cat_arr[i:end_i]), axis=1)

        df = pd.DataFrame(arr)
        df.loc[:, 0] = df.loc[:, 0].astype(int)
        df = df.astype(str)

        for idx, col in enumerate(df.columns):
            if idx == 0:
                continue
            df[col] = f'{idx}:' + df[col]
        df.to_csv(writePath, sep=' ', index=False, header=False, mode='a')


def run_experiment(train, test, use_gpu, incremental, batch_size, train_encoder=True, train_pca=True, save_data=True):
    print(f'\nbatch size: {batch_size}, n_components: {n_components}, use gpu: {use_gpu}, incremental: {incremental}')
    preprocessor = Preprocessor(n_components=n_components, batch_size=batch_size, use_gpu=use_gpu,
                                incremental=incremental)

    if train_encoder:
        print('\ntrain encoder')
        timer.start()
        preprocessor.train_one_hot_encoding(train[cat_cols_mapper.keys()])
        timer.stop()

    print('\nencoding training data')
    timer.start()
    encoded_cat_arr = preprocessor.one_hot_encoding(train[cat_cols_mapper.keys()])
    timer.stop()

    if train_pca:
        print('\ntrain PCA')
        timer.start()
        preprocessor.train_pca(encoded_cat_arr)
        timer.stop()

    print('\nrun PCA on training data')
    timer.start()
    reduced_cat_arr = preprocessor.run_pca(encoded_cat_arr)
    timer.stop()

    train = train.drop(cat_cols_to_drop, axis=1)
    print('\ndf columns after dropping cat cols: ', train.columns)

    output_columns = ['is_delayed', 'expected_duration', 'distance', 'is_holiday'] + [f'feat_{i}' for i in
                                                                                      range(n_components)]
    parquet_write_path = f'data/gbm_input/train_gpu_{use_gpu}_incremental_{incremental}_bs_{batch_size}.gzip'

    if save_data:
        print('\nsave to parquet')
        arr = np.concatenate((train.values[:, 0], train.values[:, 3:], reduced_cat_arr), axis=1)
        train_gbm_input = pd.DataFrame(arr, columns=output_columns)
        train_gbm_input.to_parquet(parquet_write_path,
                                   index=False,
                                   compression='gzip')

        writePath = f'data/tgbm_input/train_gpu_{use_gpu}_incremental_{incremental}_bs_{batch_size}.txt'
        to_tgbm_input(train, reduced_cat_arr, writePath)

    print('encoding test data')
    timer.start()
    encoded_cat_arr = preprocessor.one_hot_encoding(test[cat_cols_mapper.keys()])
    timer.stop()

    print('run PCA on test data')
    timer.start()
    reduced_cat_arr = preprocessor.run_pca(encoded_cat_arr)
    timer.stop()

    test = test.drop(cat_cols_to_drop, axis=1)
    print('df columns after dropping cat cols: ', test.columns)

    if save_data:
        parquet_write_path = f'data/gbm_input/test_gpu_{use_gpu}_incremental_{incremental}_bs_{batch_size}.gzip'

        print(f'\nsave to parquet {parquet_write_path}')
        timer.start()
        arr = np.concatenate((test.values, reduced_cat_arr), axis=1)
        test_gbm_input = pd.DataFrame(arr, columns=output_columns)
        test_gbm_input.to_parquet(parquet_write_path,
                                  index=False,
                                  compression='gzip')
        timer.stop()

        tgbm_writePath = f'data/tgbm_input/test_gpu_{use_gpu}_incremental_{incremental}_bs_{batch_size}.txt'
        print(f'\nsave to txt file {tgbm_writePath}')
        to_tgbm_input(test_gbm_input, reduced_cat_arr, tgbm_writePath)


if __name__ == "__main__":

    fpath_list = ['data/processed/' + fname for fname in os.listdir('data/processed')]

    columns = [
        'is_delayed',
        'Origin_iata_idx',
        'Dest_iata_idx',
        'Year',
        'Quarter',
        'Month',
        'DayOfWeek',
        'IATA_CODE_Reporting_Airline',
        'CRSDepTime',
        'CRSArrTime',
        'CRSElapsedTime',
        'Distance',
        'is_holiday',
    ]

    int_cols = ['is_delayed', 'is_holiday']

    cat_cols_mapper = {
        'Year': str,
        'Quarter': str,
        'Month': str,
        'DayOfWeek': str,
        'IATA_CODE_Reporting_Airline': str,
        'Origin_iata_idx': str,
        'Dest_iata_idx': str,
        'CRSDepTime': str,
        'CRSArrTime': str
    }

    cat_cols_to_drop = [
        'Year',
        'Quarter',
        'Month',
        'DayOfWeek',
        'IATA_CODE_Reporting_Airline',
        'CRSDepTime',
        'CRSArrTime',
    ]

    timer = Timer()

    raw_train_path = 'data/raw/raw_train.parquet.gzip'
    raw_test_path = 'data/raw/raw_test.parquet.gzip'

    print('load data')
    timer.start()
    if os.path.isfile(raw_train_path) and os.path.isfile(raw_test_path):
        print('load from parquet')
        train = pd.read_parquet(raw_train_path)
        test = pd.read_parquet(raw_test_path)
    else:
        print('load from csv')
        train, test = load_flight_data(raw_train_path, raw_test_path, is_save_data=True)
        timer.stop()

    print(train.columns)

    batch_size = 50000
    n_components = 50
    use_gpu = True
    incremental = False

    run_experiment(train, test, use_gpu=use_gpu, incremental=incremental, batch_size=batch_size, train_pca=True,
                   train_encoder=False)
