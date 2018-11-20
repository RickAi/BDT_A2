from __future__ import print_function
import numpy as np
from numpy.random import RandomState
import copy
import pandas as pd


class PMF():

    def __init__(self, train_data, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=50, momuntum=0.8,
                 lr=0.001, iters=200, seed=None):
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.momuntum = momuntum
        self.R = self.load_matrix(train_data)
        self.random_state = RandomState(seed)
        self.iterations = iters
        self.lr = lr
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1

        U = 0.1 * self.random_state.rand(np.size(self.R, 0), latent_size)
        self.U = pd.DataFrame(U, index=self.R.index.values, columns=np.arange(latent_size))
        V = 0.1 * self.random_state.rand(np.size(self.R, 1), latent_size)
        self.V = pd.DataFrame(V, index=self.R.columns.values, columns=np.arange(latent_size))

    def load_matrix(self, train_data):
        user_set = set(train_data[:, 0])
        movie_set = set(train_data[:, 1])
        user_count = len(user_set)
        movie_count = len(movie_set)
        print('user count: %d, movie count: %d' % (user_count, movie_count))
        # np.zeros((user_count, movie_count)),
        R = pd.DataFrame(index=user_set, columns=movie_set)
        print('load_matrix success...')
        for item in train_data:
            R.at[item[0], item[1]] = item[2]
        return R

    # the loss function of the model
    def loss(self):
        loss = np.sum(self.I * (self.R - np.dot(self.U, self.V.T)) ** 2) + self.lambda_alpha * np.sum(
            np.square(self.U)) + self.lambda_beta * np.sum(np.square(self.V))
        return loss

    def RMSE(self, preds, truth):
        return np.sqrt(np.mean(np.square(preds - truth)))

    def predict(self, data):
        index_data = np.array([[item[0], item[1]] for item in data])
        u_features = self.U.reindex(index_data.take(0, axis=1), fill_value=0).values
        v_features = self.V.reindex(index_data.take(1, axis=1), fill_value=0).values
        preds_value_array = np.sum(u_features * v_features, 1)
        return preds_value_array

    def train(self, vali_data=None):
        train_loss_list = []
        vali_rmse_list = []
        last_vali_rmse = None

        # monemtum
        momuntum_u = np.zeros(self.U.shape)
        momuntum_v = np.zeros(self.V.shape)

        for it in range(self.iterations):
            print('start traning iteration:{: d}'.format(it))

            # derivate of Vi
            grads_u = np.dot(self.I * (self.R - np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha * self.U

            # derivate of Tj
            grads_v = np.dot((self.I * (self.R - np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta * self.V

            # update the parameters
            momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
            momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
            self.U = self.U - momuntum_u
            self.V = self.V - momuntum_v

            # training evaluation
            train_loss = self.loss()
            train_loss_list.append(train_loss)

            vali_preds = self.predict(vali_data)
            vali_rmse = self.RMSE(vali_preds, np.asarray(vali_data[:, 2], dtype=float))
            vali_rmse_list.append(vali_rmse)

            # print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, vali_rmse))
            print('traning iteration:{: d} , vali_rmse:{: f}'.format(it, vali_rmse))
            break

            # if last_vali_rmse and (last_vali_rmse - vali_rmse) <= 0:
            #     print('convergence at iterations:{: d}'.format(it))
            #     break
            # else:
            #     last_vali_rmse = vali_rmse

        return self.U, self.V, train_loss_list, vali_rmse_list
