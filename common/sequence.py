# coding: utf-8
import sys
sys.path.append('..')
import numpy as np


id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char

# 데이터셋 split 함수
def load_data(x_data, t_data, valid_ratio=0.1, test_ratio=0.1, seed=1984):

    # 뒤섞기
    np.random.seed(seed)
    indices = np.arange(len(x_data))
    np.random.shuffle(indices)
    x_data = np.array(x_data)[indices]
    t_data = np.array(t_data)[indices]

    # 검증 및 테스트 데이터 크기 계산
    total_size = len(x_data)
    test_size = int(total_size * test_ratio)
    valid_size = int(total_size * valid_ratio)

    # 데이터셋 분리
    x_train = x_data[: total_size - test_size - valid_size]
    t_train = t_data[: total_size - test_size - valid_size]
    x_valid = x_data[total_size - test_size - valid_size : total_size - test_size]
    t_valid = t_data[total_size - test_size - valid_size : total_size - test_size]
    x_test = x_data[total_size - test_size :]
    t_test = t_data[total_size - test_size :]

    return (x_train, t_train), (x_valid, t_valid), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char