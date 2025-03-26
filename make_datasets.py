import pandas as pd
import numpy as np
import os
import argparse

from utils import DATA_INFO

np.random.seed(0)  # 데이터셋 분할 시 항상 동일한 값 나오게 고정

def get_args():
    """
    argmument 받는 함수
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='기존 데이터 경로')
    parser.add_argument('--data_model_path', type=str, default='./data_model', help='실제 모델에 사용할 데이터 경로')
    parser.add_argument('--multiples', type=int, default=6, help='합성 데이터 최대 배수')

    args = parser.parse_args()

    return args

def load_single_dataset(data, args):
    """
    단일 dataset (original, synthetic, cols_info) CSV 파일을 로드하고,
    불필요 열 제거와 cirrhosis 라벨 변환(필요할 경우) 등 전처리를 수행한다.
    """
    # CSV 경로 지정
    original_path = os.path.join(args.data_path, 'original_data', f'original_{data.name}.csv')
    synthetic_path = os.path.join(args.data_path, 'synthetic_data', f'synthetic_{data.name}.csv')
    cols_info_path = os.path.join(args.data_path, 'cols_info', f'{data.name}_columns.csv')

    # CSV 파일 DataFrame 변환
    original_df = pd.read_csv(original_path)
    synthetic_df = pd.read_csv(synthetic_path)

    # original / synthetic 구분 열 추가
    original_df['type'] = 'original'
    synthetic_df['type'] = 'synthetic'

    # 제외할 열 제거
    if data.exclusion is not None:
        original_df.drop(data.exclusion, axis=1, inplace=True)
        synthetic_df.drop(data.exclusion, axis=1, inplace=True)

    # cirrhosis 데이터 특수 전처리
    if data.name == 'cirrhosis':
        original_df['Status'] = original_df['Status'].map({1: 0, 3: 0, 2: 1})
        synthetic_df['Status'] = synthetic_df['Status'].map({1: 0, 3: 0, 2: 1})

    return original_df, synthetic_df


def split_original_data(original_df):
    """
    original 데이터를 홀수라면 test가 1개 더 많도록 50:50으로 split 후,
    train / test 구분열(Set)을 반환한다.
    """
    n_data = len(original_df)
    n_test = (n_data + 1) // 2  # 홀수일 때 test가 1개 더 많게
    data_idx = np.random.permutation(n_data)  # 무작위 순서

    original_df['Set'] = 'train'
    original_df.loc[data_idx[:n_test], 'Set'] = 'test'
    
    return original_df


def save_datasets_for_multiples(original_df, synthetic_df, data, args):
    """
    original의 train 부분과 synthetic 데이터를 1배, 2배, ... multiples배까지 합쳐
    CSV로 저장한다. test는 별도로 처리.
    """
    # 저장 경로 생성
    save_path = os.path.join(args.data_model_path, data.name)
    os.makedirs(save_path, exist_ok=True)

    # test dataset 추출 후 저장
    test_df = original_df[original_df['Set'] == 'test']
    test_df.to_csv(os.path.join(save_path, f"{data.name}_original_test.csv"), index=False)

    # train dataset 분리
    train_df = original_df[original_df['Set'] == 'train']
    n_train = len(train_df)

    for multiple in range(0, args.multiples + 1):
        n_synthetic = n_train * multiple  # synthetic에서 가져올 개수
        # multiple = 0이면 synthetic 0개, 그 외는 slice
        syn_df = synthetic_df[:n_synthetic]

        # original train + synthetic
        merged_train = pd.concat([train_df, syn_df], axis=0)

        if multiple == 0:
            # original train만 저장
            merged_train.to_csv(os.path.join(save_path, f"{data.name}_original_train.csv"), index=False)
        else:
            # synthetic만 저장
            syn_df.to_csv(os.path.join(save_path, f"{data.name}_{n_synthetic:04d}_train.csv"), index=False)
            # original + synthetic 전체 저장
            merged_train.to_csv(os.path.join(save_path, f"{data.name}_{n_synthetic:04d}_with_train.csv"), index=False)


def make_datasets(args):
    """
    DATA_INFO에서 dataset 정보를 가져와, 
    각 original / synthetic CSV를 불러오고,
    50:50 split + synthetic data multiple 합산하여 
    최종 CSV 파일들을 저장한다.
    """
    # DATA_INFO를 순회하며 처리
    for data in DATA_INFO:
        # 1) 데이터 로드 + 전처리
        original_df, synthetic_df = load_single_dataset(data, args)

        # 2) original data split (train / test)
        original_df = split_original_data(original_df)

        # 3) synthetic은 전체 train 취급
        synthetic_df['Set'] = 'train'

        # 4) multiple 별로 나눠 CSV 저장
        save_datasets_for_multiples(original_df, synthetic_df, data, args)


if __name__ == '__main__':
    # 디버깅/테스트용 파라미터 세팅
    args = get_args()
    make_datasets(args)
