# XG 부스트 텍스트 유사도 분석 모델
# 데이터의 주어진 두 질문 문장 사이의 유사도를 측정해서 두 질문이 중복인지 아닌지를 판단할 수 있게 만든다.

import numpy as np
import pandas as pd

DATA_IN_PATH = 'C:/Users/0105l/Desktop/pycham/data_in/'

TRAIN_Q1_DATA_FILE = 'train_q1.npy'
TRAIN_Q2_DATA_FILE = 'train_q2.npy'
TRAIN_LABEL_DATA_FILE = 'train_label.npy'

# 훈련 데이터를 가져온다.
train_q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
train_q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
train_labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))

# 하나의 질문 쌍으로 만든다.
train_input = np.stack((train_q1_data, train_q2_data), axis=1)
print(train_input.shape)

# 학습 데이터의 일부를 모델 검증을 위한 검증 데이터로 만들어 두기.
# 훈련 set과 평가 set으로 나누기
# 전체 데이터의 20%
from sklearn.model_selection import train_test_split
train_input, eval_input, train_label, eval_label = train_test_split(train_input, train_labels, test_size=0.2, random_state=4242)


# 모델 구성
import xgboost as xgb

# xgb 부스트 모델을 사용하기 위해서는 입력값을 DMatrix 형태로 만들어야 함.
train_data = xgb.DMatrix(train_input.sum(axis=1), label=train_label) # 학습 데이터 읽어오기
eval_data = xgb.DMatrix(eval_input.sum(axis=1), label=eval_label) # 평가 데이터 읽어오기

# 학습 데이터와 검증 데이터 -> 각 상태의 문자열과 함께 튜플 형태로 구성한다.
data_list = [(train_data, 'train'), (eval_data, 'valid')]

# 모델 생성 및 학습 진행
params = {} # 인자를 통해 xgb 모델을 넣어 주자
params['objective'] = 'binary:logistic'  # 목적함수 : 이진 로지스틱 함수
params['eval_metric'] = 'rmse' # 평가지표 사용: rmse(root mean squared error)

bst = xgb.train(params, train_data, num_boost_round = 1000, evals = data_list, early_stopping_rounds=10)
# num_boost_round : 데이터를 반복하는 횟수(에폭을 의미)
# evals : 모델 검증 시 사용할 전체 데이터 쌍
# early_stopping_rounds : 조기 멈춤 횟수 -> 그 횟수 동안 에러값이 별로 줄어들지 않으면 학습을 조기에 멈추도록 함.

# 테스트 데이터 가져오기
# 전처리한 평가 데이터 불러오기
TEST_Q1_DATA_FILE = 'test_q1.npy'
TEST_Q2_DATA_FILE = 'test_q2.npy'
TEST_ID_DATA_FILE = 'test_id.npy'

test_q1_data = np.load(open(DATA_IN_PATH + TEST_Q1_DATA_FILE, 'rb'))
test_q2_data = np.load(open(DATA_IN_PATH + TEST_Q2_DATA_FILE, 'rb'))
test_id_data = np.load(open(DATA_IN_PATH + TEST_ID_DATA_FILE, 'rb'), allow_pickle=True)

# 예측하기
# XG 부스트 모델에 적용할 수 있게끔 형식에 맞춰 만든 후 모델의 predict 함수에 적용
test_input = np.stack((test_q1_data, test_q2_data), axis=1)
test_data = xgb.DMatrix(test_input.sum(axis=1))
test_predict = bst.predict(test_data)

# 평가데이터의 id값과 예측값 => 하나의 데이터프레임으로 만든 후 csv파일로 저장
import os

DATA_OUT_PATH = 'C:/Users/0105l/Desktop/pycham/data_in/'
if not os.path.exists(DATA_OUT_PATH):
    os.makedirs(DATA_OUT_PATH)

output = pd.DataFrame({'test_id': test_id_data, 'is_duplicate' : test_predict})
output.to_csv(DATA_OUT_PATH + 'simple2_xgb.csv', index=False)
