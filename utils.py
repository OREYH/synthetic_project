import torch
from collections import namedtuple

Dataset = namedtuple('Dataset', ['name', 'target', 'exclusion']) # namedtuple로 데이터 관리

# 데이터 이름
DATA_NAME = ('breast_cancer', 'thyroid_cancer', 'heart_disease', 'heart_failure', 'diabetes', 'cirrhosis')
# 데이터 종속 변수(target)
DATA_TARGET = ('diagnosis', 'Recurred', 'target', 'HeartDisease', 'class', 'Status')
# 데이터 제외할 열
DATA_EXCLUSION = (None, None, None, None, None, 'N_Days')

# 데이터셋의 정보를 담고 있는 list
DATA_INFO = []
for name, target, exclusion in zip(DATA_NAME, DATA_TARGET, DATA_EXCLUSION):
    DATA_INFO.append(Dataset(name, target, exclusion))


def get_saint_scheduler(scheduler_name, epochs, optimizer):
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    elif scheduler_name == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                      milestones=[epochs // 2.667, epochs // 1.6, epochs // 1.142], gamma=0.1)
    return scheduler

if __name__ == '__main__':
    data_name = 'breast_cancer'


    for item in DATA_INFO:
        if item.name == data_name:
            target = item.target
    #target = [True if item.name == data_name else False for item in DATA_INFO]
