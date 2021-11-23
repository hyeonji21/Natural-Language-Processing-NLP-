import json
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers

# 학습 데이터 불러오기
DATA_IN_PATH = 'C:/Users/0105l/Desktop/pycham/자연어처리/data_in/'
DATA_OUT_PATH = 'C:/Users/0105l/Desktop/pycham/자연어처리/data_out/'

TRAIN_Q1_DATA_FILE = 'train_q1.npy'
TRAIN_Q2_DATA_FILE = 'train_q2.npy'
TRAIN_LABEL_DATA_FILE = 'train_label.npy'
DATA_CONFIGS = 'data_configs.json'

# 훈련 데이터를 가져온다.
q1_data = np.load(open(DATA_IN_PATH + TRAIN_Q1_DATA_FILE, 'rb'))
q2_data = np.load(open(DATA_IN_PATH + TRAIN_Q2_DATA_FILE, 'rb'))
labels = np.load(open(DATA_IN_PATH + TRAIN_LABEL_DATA_FILE, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH + DATA_CONFIGS, 'r'))

# 모듈 구현
class SentenceEmbedding(layers.Layer):
    def __init__(self, **kargs):
        super(SentenceEmbedding, self).__init__()

        self.conv = layers.Conv1D(kargs['conv_num_filters'], kargs['conv_window_size'],
                                  activation=tf.keras.activations.relu,  #활성화 함수 : relu 함수
                                  padding='same') #패딩 방법 : 입력값과 출력에 대한 크기를 동일하게 하기 위해 'same'의 값으로 설정
        self.max_pool = layers.MaxPool1D(kargs['max_pool_seq_len'], 1) #풀링 영역에 대한 크기 정의 (pool_size : 1)
        self.dense = layers.Dense(kargs['sent_embedding_dimension'], #Dense 레이어 : 문장 임베딩 벡터로 출력한 차원 수 정의
                                  activation=tf.keras.activations.relu)


    def call(self, x):  # 위에서 정의한 모듈들을 call 함수에서 호출해 모듈 연산 과정을 정의
        x = self.conv(x) #합성곱 레이어
        x = self.max_pool(x) #맥스풀 레이어
        x = self.dense(x) #뽑은 특징값의 차원수를 조절하기 위한 dense 레이어

        return tf.squeeze(x, 1) #불필요한 차원을 제거하기 위해 squeeze 함수 적용


# 문장 임베딩 모듈을 모델에 적용
# Dense 레이어 : 두 base와 hypothesis 문장에 대한 유사도를 계산하기 위해 만들어진 레이어
class SentenceSimilarityModel(tf.keras.Model):
    def __init__(self, **kargs):
        super(SentenceSimilarityModel, self).__init__(name=kargs['model_name'])

        self.word_embedding = layers.Embedding(kargs['vocab_size'],
                                               kargs['word_embedding_dimension'])
        self.base_encoder = SentenceEmbedding(**kargs)
        self.hypo_encoder = SentenceEmbedding(**kargs)
        self.dense = layers.Dense(kargs['hidden_dimension'], # 레이어 -> 두 문장간의 관계를 표현할 수 있는 벡터 출력
                                  activation=tf.keras.activations.relu)
        self.logit = layers.Dense(1, activation=tf.keras.activations.sigmoid) #sigmoid 함수 -> 유사성을 하나의 값으로 표현할 수 있게 하고,
        self.dropout = layers.Dropout(kargs['dropout_rate'])                  # 출력값의 범위를 0~1로 표현하기 위해 활성화 함수로 지정
                              # 드롭아웃 레이어 생성
    def call(self, x):
        x1, x2 = x
        b_x = self.word_embedding(x1)
        h_x = self.word_embedding(x2)
        b_x = self.dropout(b_x)
        h_x = self.dropout(h_x)

        b_x = self.base_encoder(b_x)
        h_x = self.hypo_encoder(h_x)

        e_x = tf.concat([b_x, h_x], -1)
        e_x = self.dense(e_x)
        e_x = self.dropout(e_x)

        return self.logit(e_x)


# 모델 하이퍼파라미터 정의
model_name = 'cnn_similarity'
BATCH_SIZE = 1024
NUM_EPOCHS = 100
VALID_SPLIT = 0.1
MAX_LEN = 31

kargs = {'model_name':model_name,
         'vocab_size': prepro_configs['vocab_size'], # vocab_size, word_embedding_dimension : 단어 임베딩을 위한 차원 값
         'word_embedding_dimension':100,             # 이 두 설정값은 워드 임베딩에서 활용용
        'conv_num_filters':300,  #합성곱 레이어를 위한 차원값
         'conv_window_size':3,   #합성곱 레이어를 위한 윈도우 크기
         'max_pool_seq_len':MAX_LEN, #맥스 풀링을 위한 고정 길이
         'sent_embedding_dimension':128, #문장 임베딩에 대한 차원값
         'dropout_rate':0.2,
         'hidden_dimension':200, #Dense 레이어에 대한 차원 값
         'output_dimension':1}

# 모델 생성
Model = SentenceSimilarityModel(**kargs)

Model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), # 옵티마이저 : 아담 사용
              loss=tf.keras.losses.BinaryCrossentropy(), # 손실함수 : 이진 교차 엔트로피 함수 사용
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]) # 평가 : 중복을 예측한 것에 대한 정확도 측정

# 모델 생성
# 오버피팅을 막기 위한 ealrystop 추가
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=1)
# min_delta : the threshold that triggers the termination (acc should at least improve 0.001)
# 종료를 trigger 하는 임계값
# patience : no improvement epochs (patience = 1, 1번 이상 상승이 없으면 종료)

checkpoint_path = DATA_OUT_PATH + model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create path if exists
if os.path.exists(checkpoint_dir):
    print("{} -- Folder already exists \n".format(checkpoint_dir))
else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("{} -- Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

# 두 개의 문장 벡터를 입력하는 것이기 때문에 모델에 입력하는 값이 두 개라는 점을 확인하기.
# 모델에 입력하는 값은 튜플로 구성해서 입력
history = Model.fit((q1_data, q2_data), labels, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])

# 정의한 에폭만큼 데이터를 대상으로 모델을 학습 및 검증
plot_graphs(history, 'loss')
plot_graphs(history, 'accuracy')

# 데이터 평가
TEST_Q1_DATA_FILE = 'test_q1.npy'
TEST_Q2_DATA_FILE = 'test_q2.npy'
TEST_ID_DATA_FILE = 'test_id.npy'

test_q1_data = np.load(open(DATA_IN_PATH + TEST_Q1_DATA_FILE, 'rb'))
test_q2_data = np.load(open(DATA_IN_PATH + TEST_Q2_DATA_FILE, 'rb'))
test_id_data = np.load(open(DATA_IN_PATH + TEST_ID_DATA_FILE, 'rb'), allow_pickle=True)

SAVE_FILE_NM ='weight.h5'
Model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))

predictions = Model.predict((test_q1_data, test_q2_data), batch_size=BATCH_SIZE)
predictions = predictions.squeeze(-1)

output = pd.DataFrame(data={"test_id":test_id_data, "is_duplicate": list(predictions)})
output.to_csv("cnn_predict.csv", index=False, quoting=3)
