"""
train.py
설명: 다양한 모델(RandomForest, XGBoost, LightGBM, CatBoost, TabNet, SAINT)을
    이용해 지정된 CSV 데이터를 학습/평가하고, 결과를 저장하는 메인 스크립트.
    - 데이터셋을 불러오고 전처리
    - 모델별 학습 과정을 거친 뒤
    - 반복(iterations) 수행 결과의 평균 성능을 CSV로 저장
"""

import argparse
import os
import glob
import copy
import yaml
import warnings
import importlib
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import namedtuple

# Torch / Sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# 기타 파일
from datasets import TabularDataset
from utils import DATA_INFO, get_saint_scheduler
from model.SAINT.data_healthcare import data_preprocessor, MyDatasetCatCon
from model.SAINT.models import SAINT
from model.SAINT.augmentations import embed_data_mask

warnings.filterwarnings("ignore", message="No early stopping will be performed*")

# --------------------------------------------------------------------------------
# 1. 유틸 함수
# --------------------------------------------------------------------------------
def get_func_from_string(func_str: str):
    """
    문자열로 된 함수 경로(예: 'torch.optim.Adam')를 모듈로부터 불러와 반환.
    """
    module_path, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)

def get_all_train_csv_paths(args):
    """
    args.data_model_path 내부에서 '_train.csv' 로 끝나는 모든 파일을 찾고,
    args.data_name에 지정된 이름의 폴더만 필터링해서 반환.
    """
    csv_path_list = glob.glob(os.path.join(args.data_model_path, '**', '*_train.csv'), recursive=True)
    csv_path_list = [path for path in csv_path_list if path.split(os.sep)[-2] in args.data_name]
    return csv_path_list

def make_csv_path_dict(csv_path_list):
    """
    DATA_INFO에 명시된 각 데이터셋 이름별로 csv_path를 묶어서 dict로 반환.
    예: { 'breast_cancer': [csv1, csv2], 'cirrhosis': [csv3, csv4], ... }
    """
    csv_path_dict = {item.name: [] for item in DATA_INFO}
    for csv_path in csv_path_list:
        data_name = csv_path.split(os.sep)[2]
        csv_path_dict[data_name].append(csv_path)
    return csv_path_dict

def get_config(args):
    """
    config_path 내부에서 '{args.model}_config.yaml' 파일을 읽어와
    yaml.full_load 후 딕셔너리를 반환.
    """
    cfg_file = os.path.join(args.config_path, f"{args.model}_config.yaml")
    with open(cfg_file, "r") as file:
        config = yaml.full_load(file)
    return config

# --------------------------------------------------------------------------------
# 2. TabNet 관련 함수
# --------------------------------------------------------------------------------
def build_tabnet_model(config, cat_idxs, cat_dims, df_len):
    """
    TabNet 모델을 생성하기 전, config와 train 파라미터를 조정해주는 함수.
    - cat_idxs, cat_dims: 범주형 인덱스/차원 정보
    - df_len: 데이터셋 크기(steps_per_epoch 계산용)
    """
    model_params = copy.deepcopy(config['model_params'])
    train_params = copy.deepcopy(config['train_params'])

    # random_state -> seed로 rename
    model_params['seed'] = model_params.pop('random_state', None)

    # cat feature 정보
    model_params['cat_idxs'] = cat_idxs
    model_params['cat_dims'] = cat_dims

    # optimizer, scheduler 함수명을 실제 함수 객체로 변환
    if isinstance(model_params.get('optimizer_fn'), str):
        model_params['optimizer_fn'] = get_func_from_string(model_params['optimizer_fn'])
    if isinstance(model_params.get('scheduler_fn'), str):
        model_params['scheduler_fn'] = get_func_from_string(model_params['scheduler_fn'])

    # steps_per_epoch, epochs 설정
    scheduler_params = copy.deepcopy(model_params.get('scheduler_params', {}))
    scheduler_params['steps_per_epoch'] = int(df_len / train_params['batch_size']) + 1
    scheduler_params['epochs'] = train_params['max_epochs']
    model_params['scheduler_params'] = scheduler_params

    # 모델 생성
    model = TabNetClassifier(**model_params)
    return model

def train_tabnet_model(config, model, X_train, y_train):
    """
    TabNet 모델에 대해 config['train_params']를 적용하여
    model.fit()을 수행.
    """
    train_params = copy.deepcopy(config['train_params'])
    model.fit(X_train=X_train.values, y_train=y_train.values, **train_params)

# --------------------------------------------------------------------------------
# 3. SAINT 관련 함수
# --------------------------------------------------------------------------------
def train_saint_model(config, cat_dims, cat_idxs, con_idxs, X_train, X_test, y_train, y_test):
    """
    SAINT 모델을 config 정보를 바탕으로 초기화하고 학습한 뒤,
    평가용 test_loader와 함께 model, device, task를 반환한다.
    """
    cfg_copy = copy.deepcopy(config)
    model_params = cfg_copy['model_params']
    train_params = cfg_copy['train_params']

    # random_state -> seed
    model_params['seed'] = model_params.pop('random_state', None)
    
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(model_params['seed'])

    # 데이터 전처리 (cat_dims, cat_idxs, con_idxs, X_train...)
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std = \
        data_preprocessor(X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs)

    # dataloader 생성
    continuous_mean_std = np.array([train_mean, train_std], dtype=np.float32)
    train_ds = MyDatasetCatCon(X_train, y_train, cat_idxs, continuous_mean_std)
    test_ds  = MyDatasetCatCon(X_test,  y_test,  cat_idxs, continuous_mean_std)
    train_loader = DataLoader(train_ds, batch_size=train_params['batch_size'])
    test_loader  = DataLoader(test_ds,  batch_size=train_params['batch_size'])

    # 라벨 차원
    y_dim = len(np.unique(y_train['data'][:, 0]))

    # CLS Token을 위해 cat_dims에 1 추가
    cat_dims_mod = np.append([1], cat_dims).astype(int)

    # 모델 생성
    model = SAINT(
        categories      = tuple(cat_dims_mod),
        num_continuous  = len(con_idxs),
        dim             = model_params['embedding_size'],
        dim_out         = 2 if (y_dim == 2 and model_params['task'] == 'binary') else 1,  # CrossEntropy vs BCE
        depth           = model_params['transformer_depth'],
        heads           = model_params['attention_heads'],
        attn_dropout    = model_params['attention_dropout'],
        ff_dropout      = model_params['ff_dropout'],
        mlp_hidden_mults= (4, 2),
        cont_embeddings = model_params['cont_embeddings'],
        attentiontype   = model_params['attentiontype'],
        final_mlp_style = model_params['final_mlp_style'],
        y_dim           = y_dim
    )
    model.to(device)

    # Loss 설정
    if y_dim == 2 and model_params['task'] == 'binary':
        criterion = nn.CrossEntropyLoss()
    elif y_dim > 2 and model_params['task'] == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    elif model_params['task'] == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Unsupported task type in SAINT config.")

    # Optimizer 설정
    lr = model_params['optimizer_params']['lr']
    if model_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = get_saint_scheduler(model_params['scheduler'], train_params['epoch'], optimizer)
    elif model_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    elif model_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = None

    # 모델 학습
    for epoch in range(train_params['epochs']):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            x_categ, x_cont, y_gts, cat_mask, con_mask = [b.to(device) for b in batch]

            # 입력 데이터 -> 임베딩 (masking 포함)
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
            
            # Transformer Forward
            reps  = model.transformer(x_categ_enc, x_cont_enc)
            y_reps= reps[:, 0, :]  # CLS Token만 취함

            # MLP 최종 예측
            y_outs= model.mlpfory(y_reps)

            # Loss 계산
            if model_params['task'] == 'regression':
                loss = criterion(y_outs, y_gts.float())
            else:
                # CrossEntropyLoss의 요구사항: y_gts는 long 타입
                loss = criterion(y_outs, y_gts.squeeze().long())

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    return model, test_loader, device, model_params['task']

@torch.no_grad()
def test_saint(model, data_loader, device, task, vision_dset=False):
    """
    학습된 SAINT 모델을 평가:
    - 이진분류면 softmax 후 마지막 로짓의 확률을 auc 계산용으로 사용
    - accuracy, auroc, precision, recall을 반환
    """
    model.eval()
    m = nn.Softmax(dim=1)

    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    y_prob = torch.empty(0).to(device)

    for batch in data_loader:
        x_categ, x_cont, y_gts, cat_mask, con_mask = [b.to(device) for b in batch]

        _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)

        # 실제 라벨, 예측 라벨, 예측 확률
        y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
        predicted = torch.argmax(y_outs, dim=1).float()
        y_pred = torch.cat([y_pred, predicted], dim=0)

        if task == 'binary':
            probs = m(y_outs)[:, -1].float()
            y_prob = torch.cat([y_prob, probs], dim=0)

    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_prob = y_prob.cpu().numpy() if y_prob.numel() > 0 else None

    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    return acc, auroc, precision, recall

# --------------------------------------------------------------------------------
# 4. 기타 모델 평가 (RF, XGB, LGBM, CatBoost, TabNet) 공통 함수
# --------------------------------------------------------------------------------
def test(args, model, X_test, y_test):
    """
    RandomForest, XGBoost, LightGBM, CatBoost, TabNet 모델 평가.
    - accuracy, auroc, precision, recall을 계산하여 반환.
    """
    # 예측
    if args.model in ['random_forest', 'xgboost', 'catboost']:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    elif args.model == 'lightgbm':
        y_prob = model.predict(X_test)
        y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]
    elif args.model == 'tabnet':
        y_prob = model.predict_proba(X_test.values)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

    # 스코어 계산
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    return acc, auroc, precision, recall

# --------------------------------------------------------------------------------
# 5. 메인 로직 (Train & Evaluate)
# --------------------------------------------------------------------------------
def train_and_evaluate(args, csv_path_dict):
    """
    주어진 csv_path_dict를 순회하며,
    - 각 CSV에 대해 TabularDataset 생성
    - 모델(선택된 args.model)에 따라 학습·평가 n번 반복
    - 결과의 평균값을 result_list에 누적
    """
    # 화면에 모델명 출력
    terminal_size = os.get_terminal_size().columns
    print(f"  {args.model}  ".center(terminal_size, "="))

    # config 로드
    config = get_config(args)
    model_params = config.get("model_params", {})

    # 결과 저장 용도
    Result = namedtuple('Result', ['csv_name', 'metrics'])
    result_list = []

    # CSV 파일 루프
    for data_name, csv_paths in csv_path_dict.items():
        for csv_path in sorted(csv_paths):
            csv_name = os.path.basename(csv_path)
            print(f"[csv]: {csv_name}")

            # 데이터셋 생성
            dataset = TabularDataset(args, csv_path)
            X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs = dataset.get_data()

            # n번 반복하여 평균 성능 계산
            avg_acc = avg_auroc = avg_precision = avg_recall = 0.0
            for idx in tqdm(range(args.iterations)):
                model_params['random_state'] = idx

                # 모델 분기
                if args.model == 'random_forest':
                    model = RandomForestClassifier(**model_params)
                    model.fit(X_train, y_train)
                    acc, auroc, precision, recall = test(args, model, X_test, y_test)

                elif args.model == 'xgboost':
                    model = XGBClassifier(**model_params)
                    model.fit(X_train, y_train)
                    acc, auroc, precision, recall = test(args, model, X_test, y_test)

                elif args.model == 'lightgbm':
                    train_dataset = lgb.Dataset(X_train, label=y_train)
                    model = lgb.train(model_params, train_dataset)
                    acc, auroc, precision, recall = test(args, model, X_test, y_test)

                elif args.model == 'catboost':
                    train_dataset = Pool(X_train, label=y_train)
                    model = CatBoostClassifier(**model_params)
                    model.fit(train_dataset)
                    acc, auroc, precision, recall = test(args, model, X_test, y_test)

                elif args.model == 'tabnet':
                    model = build_tabnet_model(config, cat_idxs, cat_dims, X_train.shape[0])
                    train_tabnet_model(config, model, X_train, y_train)
                    acc, auroc, precision, recall = test(args, model, X_test, y_test)

                elif args.model == 'saint':
                    model, test_loader, device, task = train_saint_model(
                        config, cat_dims, cat_idxs, con_idxs,
                        X_train, X_test, y_train, y_test
                    )
                    acc, auroc, precision, recall = test_saint(model, test_loader, device, task)

                # 반복 결과 누적
                avg_acc       += acc
                avg_auroc     += auroc
                avg_precision += precision
                avg_recall    += recall

            # 반복 평균
            n = float(args.iterations)
            avg_acc       /= n
            avg_auroc     /= n
            avg_precision /= n
            avg_recall    /= n

            print(f"[평균] Accuracy: {avg_acc:.4f}, AUROC: {avg_auroc:.4f}, "
                  f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}\n")
            
            # result_list에 저장
            result_list.append(Result(csv_name, (avg_acc, avg_auroc, avg_precision, avg_recall)))

    return result_list

def save_results_to_csv(result_list, result_path):
    """
    result_list(namedtuple 리스트)에 들어있는 평가 지표를
    CSV 파일로 저장.
    """
    records = []
    for res in result_list:
        acc, auroc, precision, recall = res.metrics
        records.append({
            "csv_name":   res.csv_name,
            "accuracy":   acc,
            "auroc":      auroc,
            "precision":  precision,
            "recall":     recall
        })

    df = pd.DataFrame(records)
    os.makedirs(result_path, exist_ok=True) # 경로 생성
    save_file = os.path.join(result_path, "result_metrics.csv")
    df.to_csv(save_file, index=False)
    print(f"[INFO] 평가 결과가 '{save_file}'에 저장되었습니다.")

# --------------------------------------------------------------------------------
# 6. 메인 실행
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # 디버깅/기본값 설정
    args.model = 'saint'
    args.data_path = './data'
    args.data_model_path = './data_model_'
    args.config_path = './config'
    args.checkpoint_path = './checkpoint'
    args.data_name = ['cirrhosis']
    args.iterations = 3
    args.result_path = './result'

    # 1) 지정된 경로에서 CSV 파일 찾기
    csv_list = get_all_train_csv_paths(args)
    print("사용할 train csv 파일 개수:", len(csv_list))

    # 2) CSV 경로를 dict 형태로 정리
    csv_path_dict = make_csv_path_dict(csv_list)

    # 3) 모델 학습 & 평가
    result_list = train_and_evaluate(args, csv_path_dict)

    # 4) 결과 CSV 저장
    save_results_to_csv(result_list, args.result_path)
