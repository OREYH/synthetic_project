# main.py

import argparse
import os
import copy

# train.py의 함수들을 import
from train import (
    get_all_train_csv_paths,
    make_csv_path_dict,
    train_and_evaluate,
    save_results_to_csv
)

def get_main_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, nargs='+',
        default=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'tabnet', 'saint'],
        choices=['random_forest', 'xgboost', 'lightgbm', 'catboost', 'tabnet', 'saint'],
        help='어떤 모델(들)을 사용할지 지정 (공백 구분)'
    )

    parser.add_argument('--data_path', type=str, default='./data',
        help='기존에 받은 데이터 경로')
    parser.add_argument('--data_model_path', type=str, default='./data_model',
        help='실제 모델에 사용할 데이터 경로')
    parser.add_argument('--data_name', type=str, nargs='+',
        default=['breast_cancer', 'thyroid_cancer', 'heart_disease', 'heart_failure', 'diabetes', 'cirrhosis'],
        choices=['breast_cancer', 'thyroid_cancer', 'heart_disease', 'heart_failure', 'diabetes', 'cirrhosis'],
        help='실행할 질병 리스트 (공백 구분)')
    parser.add_argument('--config_path', type=str, default='./config',
        help='모델 config file 경로')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint',
        help='checkpoints 저장 경로')
    parser.add_argument('--result_path', type=str, default='./result',
        help='test 결과 저장 경로')
    parser.add_argument('--iterations', type=int, default=100,
        help='모델을 몇 번 돌릴지 결정 (반복 횟수)')

    return parser.parse_args()

def main():
    # 1) 인자 파싱
    args = get_main_args()
    print("실행 인자:", args)

    # 2) CSV 파일들 찾기
    csv_path_list = get_all_train_csv_paths(args)
    print("사용할 train csv 파일 개수:", len(csv_path_list))

    # 3) CSV를 dict 형태로 묶기
    csv_path_dict = make_csv_path_dict(csv_path_list)

    # 4) 지정된 모든 모델에 대해 학습 & 평가 수행
    #    예: --model random_forest xgboost
    for model_name in args.model:
        # model_name별로 args 복사해서 model만 교체
        temp_args = copy.deepcopy(args)
        temp_args.model = model_name

        # train_and_evaluate 실행
        result_list = train_and_evaluate(temp_args, csv_path_dict)

        # 결과 CSV 저장 (모델 이름별로 다른 폴더/파일)
        model_result_path = os.path.join(temp_args.result_path, model_name)
        save_results_to_csv(result_list, model_result_path)

if __name__ == '__main__':
    main()