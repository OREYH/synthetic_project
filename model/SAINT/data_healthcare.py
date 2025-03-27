import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning) # FutureWarning은 끄기

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text+": {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def data_split(X, y, nan_mask):
    x_d = {
        'data': X.values,
        'mask': nan_mask.values }
    
    #  data와 mask의 크기가 동일한지 확인
    assert x_d['data'].shape == x_d['mask'].shape, "Shape of data not same as that of nan mask! "

    y_d = {
        'data': y.values.reshape(-1, 1)
    }
    
    return x_d, y_d

def data_preprocessor(X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs):
    # 결측지 마스크 지정
    temp_train = X_train.fillna("MissingValue")     # dataframe
    nan_mask_train = temp_train.ne("MissingValue")  # dataframe

    temp_test = X_test.fillna("MissingValue")       # dataframe
    nan_mask_test = temp_test.ne("MissingValue")    # dataframe

    # train / test 데이터 분리
    X_train, y_train = data_split(X_train, y_train, nan_mask_train)
    X_test, y_test = data_split(X_test, y_test, nan_mask_test)

    # train 연속형 데이터의 각각의 feature에 대해 평균 / 표준편차 구하기
    train_mean = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(axis=0)
    train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std

class MyDatasetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, continuous_mean_std=None, task='clf'):
        self.X = X['data']
        self.X_mask = X['mask']

        cat_cols = list(cat_cols)
        con_cols = list(set(np.arange(self.X.shape[1])) - set(cat_cols))

        self.X1 = self.X[:, cat_cols].astype(np.int64)           # categorical columns
        self.X2 = self.X[:, con_cols].astype(np.float32)         # numerical columns
        self.X1_mask = self.X_mask[:, cat_cols].astype(np.int64) # categorical columns
        self.X2_mask = self.X_mask[:, con_cols].astype(np.int64) # numerical columns

        if task == 'clf':
            self.y = Y['data']
        else:
            self.y = Y['data'].astype(np.float32)
        
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)

        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]

if __name__ == '__main__':
    DATA_PATH = '/home/station_05/work/Research/HealthCare/healthcare_respo/data_model_/cirrhosis/cirrhosis_0209_with_train.csv'
    RANDOM_SEED = 42
    TASK = 'clf'
    DATASPLIT = [.65, .15, .2]

    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_preprocessor(DATA_PATH, RANDOM_SEED, TASK, DATASPLIT)
