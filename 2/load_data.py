from numpy.ma import array
import numpy as np


def load_data(path='data/train.txt', ratio=0.8):
    data = []

    for line in open(path, 'r'):
        user_id, movie_id, rate = line.split('\t')
        data.append([int(user_id), int(movie_id), float(rate)])

    data = array(data)
    user_count = len(set(data[:, 0]))
    movie_count = len(set(data[:, 1]))

    np.random.shuffle(data)
    # train_set = data[0:int(len(data) * ratio)]
    test_set = data[int(len(data) * ratio):]
    return user_count, movie_count, data, test_set

def load_test_data(path='data/test.txt'):
    data = []

    for line in open(path, 'r'):
        user_id, movie_id = line.split('\t')
        data.append([int(user_id), int(movie_id)])

    data = array(data)
    return data
