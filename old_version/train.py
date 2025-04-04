"""
- 설명: 실제 학습 코드
1. CSV 파일 불러오기
2. 한 개의 CSV 파일로 데이터셋 구축 후, 1개의 모델을 100번 반복 실헒
"""

import argparse
import numpy as np
import pandas as pd
import glob, os
import yaml
import importlib
import copy
from tqdm import tqdm
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import warnings
from datasets import TabularDataset
from utils import DATA_INFO, get_saint_scheduler

#### 모델 모듈 ####
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from pytorch_tabnet.tab_model import TabNetClassifier

from model.SAINT.data_healthcare import data_preprocessor, MyDatasetCatCon
from model.SAINT.models import SAINT
from model.SAINT.augmentations import embed_data_mask

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

warnings.filterwarnings("ignore", message="No early stopping will be performed*")

def get_func_from_string(func_str):
    """
    예: 'torch.optim.Adam' -> torch.optim.Adam 객체 반환
    """
    module_path, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def get_all_train_csv_paths(args):
    '''
    모든 train csv 파일 경로 받는 함수
    '''
    csv_path_list = glob.glob(os.path.join(args.data_model_path, '**', '*_train.csv'), recursive=True)

    # 원하는 데이터셋 지정
    csv_path_list = [path for path in csv_path_list if path.split(os.sep)[-2] in args.data_name]

    return csv_path_list

def make_csv_path_dict(csv_path_list):
    '''
    모든 csv_path 경로를 받아 dict로 분류
    '''

    csv_path_dict = {}

    for item in DATA_INFO:
        csv_path_dict[item.name] = []
    
    for csv_path in csv_path_list:
        data_name = csv_path.split(os.sep)[2]
        csv_path_dict[data_name].append(csv_path)

    return csv_path_dict

def get_config(args):
    '''
    모델 config 파일을 열어서 반환하는 함수
    '''
    with open(os.path.join(args.config_path, f"{args.model}_config.yaml"), "r") as file:
        config = yaml.full_load(file)

    return config

def build_tabnet_model(config, cat_idxs, cat_dims, df_len):
    '''
    tabnet model를 생성하기 위해 config 조절
    '''
    model_params = copy.deepcopy(config['model_params'])
    model_params['seed'] = model_params.pop('random_state', None) # random_state -> seed로 변경
    model_params['cat_idxs'] = cat_idxs
    model_params['cat_dims'] = cat_dims

    train_params = copy.deepcopy(config['train_params'])
    scheduler_params = copy.deepcopy(model_params['scheduler_params'])

    # 문자열 → 함수 객체로 변환
    if isinstance(model_params['optimizer_fn'], str):
        model_params['optimizer_fn'] = get_func_from_string(model_params['optimizer_fn'])
    if isinstance(model_params['scheduler_fn'], str):
        model_params['scheduler_fn'] = get_func_from_string(model_params['scheduler_fn'])

    # steps_per_epoch, epochs는 직접 지정
    scheduler_params['steps_per_epoch'] = int(df_len / train_params['batch_size']) + 1
    scheduler_params['epochs'] = train_params['max_epochs']

    # 꼭 반영해줘야 함
    model_params['scheduler_params'] = scheduler_params
    
    model = TabNetClassifier(**model_params)

    return model

def train_saint_model(config, cat_dims, cat_idxs, con_idxs, X_train, X_test, y_train, y_test):
    config = copy.deepcopy(config) # config 복제
    model_params = config['model_params']
    model_params['seed'] = model_params.pop('random_state', None) # seed값 지정
    train_params = config['train_params']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 설정
    torch.manual_seed(model_params['seed']) # seed 설정

    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_test, y_test, train_mean, train_std = data_preprocessor(X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs)
    continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

    # train / test dataloader 생성
    train_ds = MyDatasetCatCon(X_train, y_train, cat_idxs, continuous_mean_std)
    train_loader = DataLoader(train_ds, batch_size=train_params['batch_size'])

    test_ds = MyDatasetCatCon(X_test, y_test, cat_idxs, continuous_mean_std)
    test_loader = DataLoader(test_ds, batch_size=train_params['batch_size'])

    y_dim = len(np.unique(y_train['data'][:, 0]))

    # CLS Token을 위해 1 추가
    cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

    # 모델 지정
    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(con_idxs),
        dim=model_params['embedding_size'],
        dim_out=1,
        depth=model_params['transformer_depth'],
        heads=model_params['attention_heads'],
        attn_dropout=model_params['attention_dropout'],
        ff_dropout=model_params['ff_dropout'],
        mlp_hidden_mults=(4, 2),
        cont_embeddings=model_params['cont_embeddings'],
        attentiontype=model_params['attentiontype'],
        final_mlp_style=model_params['final_mlp_style'],
        y_dim=y_dim)
    
    model.to(device) # 모델 device 설정
    
    # Loss 지정
    if y_dim == 2 and model_params['task'] == 'binary':
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2 and  model_params['task'] == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(device)
    elif model_params['task'] == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise'case not written yet'

    # Optimizer  / Scheduler 지정
    if model_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=model_params['optimizer_params']['lr'],
                            momentum=0.9, weight_decay=5e-4)
        scheduler = get_saint_scheduler(model_params['scheduler'], train_params['epoch'], optimizer)
    elif model_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=model_params['optimizer_params']['lr'])
    elif model_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=model_params['optimizer_params']['lr'])

    for epoch in range(train_params['epochs']):
        #### 모델 학습 ####
        model.train()
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
            x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # We are converting the data to embeddings in the next step
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)           
            reps = model.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:, 0, :]
            
            y_outs = model.mlpfory(y_reps)
            if model_params['task'] == 'regression':
                loss = criterion(y_outs, y_gts) 
            else:
                loss = criterion(y_outs, y_gts.squeeze()) 

            loss.backward()
            optimizer.step()
            if model_params['optimizer'] == 'SGD':
                scheduler.step()

    # print(f"[DEBUG] 모델 파라미터 합: {sum(p.sum().item() for p in model.parameters())}")

    return model, test_loader, device, model_params['task']

@torch.no_grad()
def test_saint(model, data_loader, device, task, vision_dset=False):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    y_prob = torch.empty(0).to(device)

    for i, data in enumerate(data_loader, 0):
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device),data[4].to(device)
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model, vision_dset)           
        reps = model.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:,0,:]
        y_outs = model.mlpfory(y_reps)
        y_test = torch.cat([y_test, y_gts.squeeze().float()],dim=0)
        y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()],dim=0)
        if task == 'binary':
            y_prob = torch.cat([y_prob, m(y_outs)[:,-1].float()], dim=0)
    
    y_test = y_test.cpu()
    y_pred = y_pred.cpu()
    y_prob = y_prob.cpu()
        
    # 평가
    acc = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    return acc, auroc, precision, recall


def train_tabnet_model(config, model, X_train, y_train):
    train_params = copy.deepcopy(config['train_params'])

    model.fit(
        X_train=X_train.values, y_train=y_train.values,
        **train_params)

def train_and_evaluate(args, csv_path_dict):
    terminal_size = os.get_terminal_size().columns # 터미널 가로 size
    print(f"  {args.model}  ".center(terminal_size, "="))

    config = get_config(args) # model config file 반환
    # 모델 파라미터 불러오기
    model_params = config.get("model_params", {})

    # 평가 결과를 담기 위한 namedtuple
    Result = namedtuple('Result', ['csv_name', 'metrics'])
    result_list = []

    for data_name, csv_path_list in  csv_path_dict.items():
        for csv_path in sorted(csv_path_list):
            csv_name = os.path.basename(csv_path)
            print(f"[csv]: {csv_name}")

            # csv에서 DataFrame 불러와 train, test 분할
            dataset = TabularDataset(args, csv_path)
            X_train, X_test, y_train, y_test, cat_idxs, cat_dims, con_idxs = dataset.get_data()

            # 평가 지표 초기화
            avg_acc, avg_auroc, avg_precision, avg_recall = 0.0, 0.0, 0.0, 0.0

            for idx in tqdm(range(args.iterations)):
                model_params['random_state'] = idx # random_state seed값 지정
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
                    model, test_loader, device, task = train_saint_model(config, cat_dims, cat_idxs, con_idxs, X_train, X_test, y_train, y_test)
                    acc, auroc, precision, recall = test_saint(model, test_loader, device, task)
                
                # 평가 지표 합산
                avg_acc += acc
                avg_auroc += auroc
                avg_precision += precision
                avg_recall += recall
            
            # 평가 지표 평균
            avg_acc /= args.iterations
            avg_auroc /= args.iterations
            avg_precision /= args.iterations
            avg_recall /= args.iterations
            
            print(f"[평균] Accuracy: {avg_acc:.4f}, AUROC: {avg_auroc:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}\n")

            # 평가 지표 저장
            result_list.append(Result(csv_name, (avg_acc, avg_auroc, avg_precision, avg_recall)))

    return result_list

def test(args, model, X_test, y_test):
        if args.model in ['random_forest', 'xgboost', 'catboost']:
            # 예측
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] # AUROC용 확률값

        elif args.model == 'lightgbm':
            y_prob = model.predict(X_test)
            y_pred = [1 if prob > 0.5 else 0 for prob in y_prob]

        elif args.model == 'tabnet':
            y_prob = model.predict_proba(X_test.values)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
        # 평가
        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)

        return acc, auroc, precision, recall

def save_results_to_csv(result_list, result_path):
    records = []
    for result in result_list:
        acc, auroc, precision, recall = result.metrics
        records.append({
            "csv_name": result.csv_name,
            "accuracy": acc,
            "auroc": auroc,
            "precision": precision,
            "recall": recall
        })
    
    save_path = os.path.join(result_path, "result_metrics.csv")
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"[INFO] 평가 결과가 '{save_path}'에 저장되었습니다.")


if __name__ == '__main__':

    #### 디버깅용 ####
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.model = 'saint'
    args.data_path = './data'
    args.data_model_path = './data_model_'
    args.config_path = './config'
    args.checkpoint_path = './checkpoint'
    args.data_name = ['cirrhosis']
    
    args.iterations = 3 # 몇 번 모델을 반복할지
    args.result_path = './result'

    csv_path_list = get_all_train_csv_paths(args)
    print("사용할 train csv 파일 개수:", len(csv_path_list))
    
    csv_path_dict = make_csv_path_dict(csv_path_list)
    result_list = train_and_evaluate(args, csv_path_dict)

    save_results_to_csv(result_list, args.result_path)