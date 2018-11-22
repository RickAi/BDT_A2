from numpy.ma import array
import random

def load_result_data(file_path='data/test.dat'):
    prefer = []

    for line in open(file_path, 'r'):
        user_id, movie_id = line.split('\t')
        prefer.append([str(user_id), str(movie_id)])

    data = array(prefer)
    return data

def load_and_split(file_path='data/train.dat'):
    data = load_data(file_path)
    return train_test_split(data)


def mapping(user_ids):
    ids_set = set(user_ids)
    count = len(ids_set)
    map = dict(zip(ids_set, range(0, count)))
    return map


def load_data(file_path):
    prefer = []

    for line in open(file_path, 'r'):
        user_id, movie_id, rating = line.split('\t')
        prefer.append([str(user_id), str(movie_id), float(rating)])

    data = array(prefer)
    return data


def train_test_split(data, ratio=0.2):
    train_data = []
    test_data = []

    for line in data:
        rand = random.random()
        if rand < ratio:
            test_data.append(line)
        else:
            train_data.append(line)

    train_data = array(train_data)
    test_data = array(test_data)
    return data, train_data, test_data
