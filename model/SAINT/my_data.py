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

def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    # data와 mask의 크기가 동일한지 확인
    assert x_d['data'].shape == x_d['mask'].shape, "Shape of data not same as that of nan mask! "

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }

    return x_d, y_d

def data_preprocessor(data_path, random_seed, task, datasplit=[.8, 0, .2]):
    np.random.seed(random_seed) # data 분할 시 seed
    
    # 1) 데이터 불러오기 
    df = pd.read_csv(data_path)

    # 2) inputs / targets 데이터 분리 
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy() # inputs
    X['Pclass'] = X['Pclass'].astype('object') # 범주형 데이터로 간주

    y = df['Survived'].values # targets

    # 3) 범주형 / 연속형 처리
    cat_indicator = np.array([True if str(dtype) == 'object' else False for dtype in X.dtypes]) # 범주형 여부
    cat_idxs = list(np.where(cat_indicator == True)[0]) # 범주형 열 인덱스
    con_idxs = list(np.where(cat_indicator != True)[0]) # 연속형 열 인덱스
    
    cat_cols = X.columns[cat_idxs].tolist()             # 범주형 열 레이블
    con_cols = X.columns[con_idxs].tolist()             # 연속형 열 레이블

    # 4) train / valid / test indices 추출
    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=X.shape[0])

    train_indices = X.loc[X["Set"] == "train"].index
    valid_indices = X.loc[X["Set"] == "valid"].index
    test_indices = X.loc[X["Set"] == "test"].index

    X = X.drop(columns=["Set"]) # Set 열 레이블 제거

    # 5) 결측치 마스크 지정
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    # 6) 범주형 데이터: 라벨 인코딩 / 연속형 데이터: 결측치 처리 / 타겟 데이터: 라벨 인코딩
    # 범주형 데이터 처리
    cat_dims = [] # 범주형 데이터 class 개수
    for col in cat_cols:
        X.loc[:, col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() # 라벨 인코더
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))

    # 연속형 데이터 처리
    for col in con_cols:
        X[col] = X[col].fillna(X.loc[train_indices, col].mean())
    
    # 타켓 데이터
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
    
    # 7) train / valid / test 데이터 분리
    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    # train 연속형 데이터의 각각의 feature에 대해 평균 / 표준편차 구하기
    train_mean = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(axis=0)
    train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(axis=0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std

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
    DATA_PATH = './data/Titanic/train.csv'
    RANDOM_SEED = 42
    TASK = 'clf'
    DATASPLIT = [.65, .15, .2]

    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_preprocessor(DATA_PATH, RANDOM_SEED, TASK, DATASPLIT)

    print(cat_dims)
    print(cat_idxs, con_idxs)
    print(train_mean, train_std)