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

if __name__ == '__main__':
    print(DATA_INFO)