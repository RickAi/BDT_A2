from load_data import *
from pmf import *

print('start loading data...')
data, train_data, test_data = load_and_split()
print('load data success...')

pmf = PMF(train_data=data, lambda_alpha=0.01, lambda_beta=0.01,
          latent_size=20, momuntum=0.9, lr=3e-5, iters=100000, seed=1)

print('start train model...')
U, V, train_loss_list, vali_rmse_list = pmf.train(vali_data=test_data)
print('train model done...')

print('start testing model.......')
preds = pmf.predict(data=test_data)
test_rmse = pmf.RMSE(preds, np.asarray(test_data[:, 2], dtype=float))
print('test rmse:{:f}'.format(test_rmse))

print('start save result.txt...')
# predict and save
result_data = load_result_data()
result_preds = pmf.predict(data=result_data)
# output the test result into result.txt
with open('result.txt', 'wb') as f:
    np.savetxt(f, result_preds, fmt='%.4f')
print('save done!')
