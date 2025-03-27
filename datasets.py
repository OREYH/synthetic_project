"""
- 설명: data_model에 저장된 csv 파일을 가지고 와, 각각의 모델에 맞게 데이터를 전처리 해주는 코드

1. 범주형 데이터 : One
"""

import argparse
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

from utils import DATA_INFO

class TabularDataset():
    ''' 모델에 맞는 1개의 데이터셋 (train / test 별도) 을 만드는 코드 '''
    def __init__(self, args, train_path):
        self.args = args   # arguments
        self.model = args.model # 모델 이름
        self.train_path = train_path # train csv 파일 경로

        self.data_name = train_path.split(os.sep)[2] # breast_cancer
        self.test_path, self.cols_info_path = self.get_csv_path() 
        self.train_df, self.test_df, self.cols_info_df = self.load_data() # 실제로 사용되는 DataFrame

        for item in DATA_INFO:
            if item.name == self.data_name:
                self.target = item.target # target 열 반환

        # 전처리 완료된 학습 데이터셋
        self.X_train, self.X_test, self.y_train, self.y_test, self.cat_idxs, self.cat_dims, self.con_idxs = self.preprocess()

    def get_csv_path(self):
        '''
        train_path를 이용해 test_path, cols_info_path 만드는 코드
        '''
        test_path = os.path.join(self.train_path[:self.train_path.rfind(self.data_name)], f"{self.data_name}_original_test.csv")
        cols_info_path = os.path.join(self.args.data_path, 'cols_info', f"{self.data_name}_columns.csv")

        return test_path, cols_info_path

    def load_data(self):
        '''
        path에서 dataframe 불러오는 코드
        '''
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        cols_info_df = pd.read_csv(self.cols_info_path)

        # cirrhosis data 예외 처리
        if self.data_name == 'cirrhosis':
            cols_info_df = cols_info_df.drop(cols_info_df[cols_info_df['열 이름'] == 'N_Days'].index)

        return train_df, test_df, cols_info_df
    
    def preprocess(self):
        '''
        데이터를 전처리하는 메소드
        '''

        # 연속형, 범주형 열 분리
        cat_cols = self.cols_info_df.loc[self.cols_info_df['구분'] == '범주형', '열 이름'].tolist()
        con_cols = self.cols_info_df.loc[self.cols_info_df['구분'] == '연속형', '열 이름'].tolist()
        cat_dims = {}

        # 범주형 데이터 처리
        for col in cat_cols:
            l_enc = LabelEncoder() # 라벨 인코더
            self.train_df.loc[:, col] = l_enc.fit_transform(self.train_df[col])
            self.test_df.loc[:, col] = l_enc.transform(self.test_df[col]) # 동일한 기준으로 인코딩
            cat_dims[col] = len(l_enc.classes_) # 범주형 데이터의 class 개수
        
        # target 열을 제외
        assert self.target in cat_cols, "target은 항상 cat_cols 안에 존재해야 해요!"
        cat_cols.remove(self.target)
        cat_dims.pop(self.target, None)


        # target, type, Set 열을 제외한 나머지 열을 독립 변수로 사용
        unused_cols = [self.target, "type", "Set"]
        X_train = self.train_df.drop(unused_cols, axis=1)
        X_test = self.test_df.drop(unused_cols, axis=1)

        cat_idxs = [X_train.columns.get_loc(col) for col in cat_cols]
        cat_dims = [cat_dims[col] for col in cat_cols]

        con_idxs = [X_train.columns.get_loc(col) for col in con_cols]
        
        # target 열을 종속 변수로 사용
        y_train = self.train_df[self.target]
        y_test = self.test_df[self.target]

        return X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs
    
    def get_data(self):
        '''
        실제 사용하는 데이터를 받는 메소드
        '''
        return self.X_train, self.X_test, self.y_train, self.y_test, self.cat_idxs, self.cat_dims, self.con_idxs

if __name__ == '__main__':
    #### 디버깅용 ####
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.model = 'xgboost'
    args.data_path = './data'
    args.data_model_path = './data_model'

    # train path / test path 분리
    train_path = './data_model/cirrhosis/cirrhosis_0209_with_train.csv'

    dataset = TabularDataset(args, train_path)
    X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs = dataset.get_data()
