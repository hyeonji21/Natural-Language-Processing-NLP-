import os #운영체제의 기능을 사용하기 위해
import re #정규표현식을 사용하기 위해
import json
import numpy as np
import pandas as pd #데이터 불러오기
from tqdm import tqdm
from konlpy.tag import Okt  #konlpy : 한글 형태소 활용

# 학습에 사용할 데이터를 위한 데이터 처리와 관련하여 몇가지 설정값을 지정한다.
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"  #어떤 의미도 없는 패딩 토큰
STD = "<SOS>"  #시작 토큰을 의미
END = "<END>"  #종료 토큰을 의미
UNK = "<UNK>"  #사전에 없는 단어를 의미
PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

# load_data 함수는 데이터를 판다스를 통해 부러오는 함수
# 판다스를 통해 데이터를 가져와 데이터프레임 형태로 만든 후 question과 answer를 돌려준다. (inputs, outputs에는 question, answer 존재)
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer


# 단어 사전을 만들기 위해 -> 데이터를 전처리한 후 단어 리스트로 먼저 만들어야함.
                    # -> 이 기능을 수행하는 data_tokenizer 함수를 먼저 정의
# 정규표현식 (re)을 사용해 특수 기호를 모두 제거, 공백 문자를 기준으로 단어들을 나눠서 전체 데이터의 모든 단어를 포함하는 단어 리스트 생성
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    return [word for word in words if word]

# prepro_like_morphlized 함수 : 한글 텍스트를 토크나이징하기 위해 형태소로 분리하는 함수
# KoNLPy에서 제공하는 Okt 형태소 분리기를 사용해 형태소 기준으로 텍스트 데이터를 토크나이징.
# 형태소로 분류한 데이터를 받아 morphs 함수를 통해 토크나이징된 리스트 객체를 받고, 이를 공백 문자 기준으로 문자열로 재구성해 반환
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ','')))
        result_data.append(morphlized_seq)

    return result_data


# make_vocabulary 함수
def make_vocabulary(vocabulary_list):
    # 리스트를 키가 단어이고 값이 인덱스인 딕셔너리를 만든다.
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    # 리스트를 키가 인덱스이고 값이 단어인 딕셔너리를 만든다.
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    # 두 개의 딕셔너리를 넘겨 준다.
    return word2idx, idx2word


# 단어 사전을 만드는 함수 정의
#  -> 경로에 단어 사전 파일이 없다면 불러와서 사용한다.
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
        if tokenize_as_morph:
            question = prepro_like_morphlized(question)
            answer = prepro_like_morphlized(answer)

        data = []
        data.extend(question)
        data.extend(answer)
        words = data_tokenizer(data)
        words = list(set(words))
        words[:0] = MARKER  # 사전에 정의한 특정 토큰들을 단어 리스트 앞에 추가한 후 마지막으로 이 리스트를 지정한 경로에 저장

        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    # word2idx : 각각 단어에 대한 인덱스 / idx2word : 인덱스에 대한 단어를 가진 딕셔너리 데이터에 해당

    return word2idx, idx2word, len(word2idx)  # 단어에 대한 인덱스 / 인덱스에 대한 단어 / 단어의 개수

# 인코더 부분 & 디코더 부분 전처리

# 인코더에 적용될 입력값을 만드는 전처리 함수
# 띄어쓰기를 기준으로 토크나이징 한다.
def enc_processing(value, dictionary, tokenize_as_morph=False):   #value:전처리할 데이터 / dictionary:단어 사전
    # 인덱스 값들을 가지고 있는 배열. (누적된다.)
    sequences_input_index = []
    # 하나의 인코딩 되는 문장의 길이를 가지고 있다. (누적된다.)
    sequences_length = []

    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    # 한줄씩 불어온다.
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence) #정규 표현식을 통해 특수문자 제거 (필터에 들어있는 값들을 ""으로 치환)
        sequence_index = [] #문장을 스페이스 단위로 자르고 있다.
        for word in sequence.split(): #잘려진 단어들이 딕셔너리에 존재하는지 보고 그 값을 가져와 sequence.index에 추가
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])  #단어 사전을 이용해 단어 인덱스로 바꿈.
            else:
                sequence_index.extend([dictionary[UNK]])  #어떤 단어가 단어 사전에 포함되어있지 않다면 UNK 토큰을 넣는다.

        #문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        #하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)을 넣어준다.
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_input_index에 넣어 준다.
        sequences_input_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경
    # -> 텐서플로우 dataset에 넣어 주기 위한 사전 작업
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_input_index), sequences_length


# 디코더의 입력값을 만드는 함수
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = [] # 인덱스 값들을 가지고 있는 배열 (누적)
    sequences_length = [] # 하나의 디코딩 입력 되는 문장의 길이를 가지고 있다. (누적)

    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        # 하나의 문장을 디코딩할 때 가지고 있기 위한 배열
        sequence_index = []
        # 문장에서 스페이스 단위별로 단어를 가져와서 딕셔너리의 값인 인덱스를 넣어 준다.
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word
                                              in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        # 하나의 문장에 길이를 넣어주고 있다.
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        # 인덱스화 되어 있는 값을 sequences_output_index에 넣어 준다.
        sequences_output_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경
    # -> 텐서플로우 dataset에 넣어 주기 위한 사전 작업
    # 넘파이 배열에 인섹드화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_output_index), sequences_length

# 디코더의 타깃값을 만드는 전처리 함수
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    # 인덱스 값들을 가지고 있는 배열 (누적)
    sequences_target_index = []
    # 형태소 토크나이징 사용 유무
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # 문장에서 스페이스 단위별로 단어를 가져와 딕셔너리의 값인 인덱스를 넣어 준다.
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        # 문장 제한 길이보다 길어질 경우 뒤에 토큰을 자르고 있다.
        # 그리고 END 토큰을 넣어 준다.
        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 인덱스화 되어 있는 값을 sequences_target_index에 넣어 준다.
        sequences_target_index.append(sequence_index)

    # 인덱스화된 일반 배열을 넘파이 배열로 변경
    #  -> 텐서플로우 dataset에 넣어 주기 위한 사전 작업
    # 넘파이 배열에 인덱스화된 배열과 그 길이를 넘겨준다.
    return np.asarray(sequences_target_index)

