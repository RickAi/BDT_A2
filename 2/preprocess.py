

userid_mapping = {}
movieid_mapping = {}

userid_counter = 0
movieid_counter = 0

train_list = []

for line in open('./data/train.dat', 'r'):
    user_id, movie_id, rate = line.split('\t')
    rate = rate.rstrip()

    if str(user_id) in userid_mapping:
        uid = int(userid_mapping[str(user_id)])
    else:
        userid_mapping[str(user_id)] = userid_counter
        uid = userid_counter
        userid_counter += 1

    if str(movie_id) in movieid_mapping:
        mid = int(movieid_mapping[str(movie_id)])
    else:
        movieid_mapping[str(movie_id)] = movieid_counter
        mid = movieid_counter
        movieid_counter += 1

    rat = float(rate)
    train_list.append((uid, mid, rat))

with open('./data/train.txt', 'w') as f:
    for item in train_list:
        f.write(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')

test_list = []
from_user_id = userid_counter
from_movie_id = movieid_counter
for line in open('./data/test.dat', 'r'):
    user_id, movie_id = line.split('\t')
    movie_id = movie_id.rstrip()

    if str(user_id) in userid_mapping:
        uid = int(userid_mapping[str(user_id)])
    else:
        print('userid: %s is not in map, convert to counter: %d' % (str(user_id), userid_counter))
        userid_mapping[str(user_id)] = userid_counter
        uid = userid_counter
        userid_counter += 1

    if str(movie_id) in movieid_mapping:
        mid = int(movieid_mapping[str(movie_id)])
    else:
        print('movieid: %s is not in map, convert to counter: %d' % (str(movie_id), movieid_counter))
        movieid_mapping[str(movie_id)] = movieid_counter
        mid = movieid_counter
        movieid_counter += 1

    test_list.append((uid, mid))

print('new user in test datasets:%d' % (userid_counter - from_user_id))
print('new movie in test datasets:%d' % (movieid_counter - from_movie_id))

with open('./data/test.txt', 'w') as f:
    for item in test_list:
        f.write(str(item[0]) + '\t' + str(item[1]) + '\n')