from load_data import *
from pmf import *

print("start read training dataset")
user_count, movie_count, train_set, test_set = load_data()
print("start read testing dataset...")
output_set = load_test_data()
loss, rmse, U, V = PMF(train=train_set, test=test_set, output_set=output_set,
                       user_count=user_count, movie_count=movie_count)
