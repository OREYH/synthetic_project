# 데이터셋 불러오기, 전처리, 분할 진행 후 저장
'''
1. original dataset 불러와 train:test = 50:50 비율로 지정
2. train original dataset의 1배, 2배, ..., 6배까지 synthetic dataset을 불러와 train data로 사용
'''
import pandas as pd
import numpy as np
import glob, os

from utils import DATA_INFO

import argparse # 디버깅용

np.random.seed(0) # 데이터셋 분할 시 항상 동일한 값 나오게 고정


#### CSV 파일 불러오는 함수 ####
def load_csv_path(args):
    original_paths = glob.glob(os.path.join(args.data_path, "original_data", "*.csv"))
    synthetic_paths = glob.glob(os.path.join(args.data_path, "synthetic_data", "*.csv"))
    info_paths = glob.glob(os.path.join(args.data_path, "info_data", "*.csv"))

    # CSV 파일 경로 정보 담고 있는 dict 생성
    path_dict = {
        'original': original_paths,
        'synthetic': synthetic_paths,
        'info': info_paths }
    
    return path_dict

#### CSV 파일에서 dataframe을 만드는 함수 ####
def make_datasets(args):
    # 데이터셋의 정보를 담고 있는 DATA_INFO 불러오기
    for data in DATA_INFO:
        # 각 csv 파일 경로 불러오기
        original_path = os.path.join(args.data_path, 'original_data', f'original_{data.name}.csv')
        synthetic_path = os.path.join(args.data_path, 'synthetic_data', f'synthetic_{data.name}.csv')
        cols_info_path = os.path.join(args.data_path, 'cols_info', f'{data.name}_columns.csv')

        # csv 파일 열어서 DataFrame 전환
        original_df = pd.read_csv(original_path)
        synthetic_df = pd.read_csv(synthetic_path)
        cols_info_df = pd.read_csv(cols_info_path)

        # 혼동 방지를 위해 original인지 synthetic인지 표기
        original_df['type'] = 'original'
        synthetic_df['type'] = 'synthetic'

        # 제외할 열이 있는지 검사 후 제거
        if data.exclusion is not None:
            original_df.drop(data.exclusion, axis=1, inplace=True)
            synthetic_df.drop(data.exclusion, axis=1, inplace=True)

        # cirrhosis 데이터에서 Status 전처리 진행
        # 1, 3 -> 0 / 2 -> 1
        if data.name == 'cirrhosis':
            original_df['Status'] = original_df['Status'].map({1:0, 3:0, 2:1})
            synthetic_df['Status'] = synthetic_df['Status'].map({1:0, 3:0, 2:1})

        # original data를 50:50으로 분리
        n_data = len(original_df)  # data 개수
        n_test = (n_data + 1) // 2 # 홀수일 경우, test가 1개 더 많도록 지정
        n_train = n_data - n_test

        data_idx = np.random.permutation(n_data) # 데이터 index 추출
        original_df['Set'] = 'train'
        original_df.loc[data_idx[:n_test], 'Set'] = 'test'

        # synthetic data는 전부 train data로만 사용
        synthetic_df['Set'] = 'train'

        # 데이터셋 저장 경로 생성
        save_path = os.path.join(args.data_save_path, data.name)
        os.makedirs(os.path.join(save_path), exist_ok=True)

        # test dataset 저장
        test_df = original_df[original_df['Set'] == 'test'] 
        test_df.to_csv(os.path.join(save_path, f"{data.name}_original_test.csv"), index=False)

        # train dataset 저장
        # original:synthetic = 1:0, 1:1, 1:2, ... , 1:args.multiples
        for multiple in range(0, args.multiples + 1):
            n_synthetic = n_train * multiple # synthetic data 개수

            # synthetic data / synthetic data + original train data
            syn_df = synthetic_df[:n_synthetic] 
            data_df = pd.concat([original_df[original_df['Set'] == 'train'], syn_df], axis=0)
            
            if multiple == 0:
                data_df.to_csv(os.path.join(save_path, f"{data.name}_original_train.csv"), index=False)
            else:
                syn_df.to_csv(os.path.join(save_path, f"{data.name}_{n_synthetic:04d}.csv"), index=False)
                data_df.to_csv(os.path.join(save_path, f"{data.name}_{n_synthetic:04d}with.csv"), index=False)


if __name__ == '__main__':
    #### 디버깅용 ####
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_path = './data'
    args.data_save_path = './data_model' # 모델이 실제로 사용하는 데이터셋 경로
    args.multiples = 6
    args.split_ratio = [0.5, 0.5]
    
    path_dict = load_csv_path(args)
    make_datasets(args)