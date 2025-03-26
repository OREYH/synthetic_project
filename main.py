from make_datasets import load_csv_path

import argparse

# Parameters 얻는 함수
def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, required=True,
            choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'tabnet', 'saint'], help='모델 지정')
    parser.add_argument('--data_path', type=str, default='./data', help='datasets 경로')
    parser.add_argument('--data_save_path', type=str, default='./model_data', help='model에 사용할 datasets 저장 경로')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint', help='checkpoints 저장 경로')
    parser.add_argument('--split_ratio', type=float, nargs=2, default=[0.5, 0.5], help='train/test 비율')
    parser.add_argument('--multiples', type=int, default=6, help='합성 데이터 최대 배수')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    print(args)
    load_csv_path(args)


if __name__ == '__main__':
    main()