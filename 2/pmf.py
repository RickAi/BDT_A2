import numpy as np


def PMF(train, test, user_count, movie_count, output_set, lr=0.005, latent=10, lambda_1=0.1, lambda_2=0.1):
    U = np.random.normal(0, 0.1, (user_count, latent))
    V = np.random.normal(0, 0.1, (movie_count, latent))
    iteration = 0
    while True:
        loss = 0.0
        for data in train:
            u = int(data[0])
            i = int(data[1])
            r = data[2]

            e = r - np.dot(U[u], V[i].T)
            U[u] = U[u] + lr * (e * V[i] - lambda_1 * U[u])
            V[i] = V[i] + lr * (e * U[u] - lambda_2 * V[i])

            loss = loss + 0.5 * (e ** 2 + lambda_1 * np.square(U[u]).sum() + lambda_2 * np.square(V[i]).sum())
        rmse = RMSE(U, V, test)

        iteration += 1
        print("The loss is: %s, the rmse is: %s at iteration: %s" % (loss, rmse, iteration))

        if iteration > 0 and iteration % 100 == 0:
            print("dump the test result in iteration:" + str(iteration))
            dump_result(U, V, output_set)


def dump_result(U, V, output_set, path='./result.txt'):
    with open(path, 'w') as f:
        for item in output_set:
            user_id = int(item[0])
            movie_id = int(item[1])
            f.write(str(predict(U, V, user_id, movie_id)) + '\n')


def predict(U, V, user_id, movie_id):
    try:
        return np.dot(U[user_id], V[movie_id].T)
    except IndexError:
        return 0.0


def RMSE(U, V, test):
    count = len(test)
    sum_rmse = 0.0
    for t in test:
        u = int(t[0])
        i = int(t[1])
        r = t[2]
        pr = predict(U, V, u, i)
        sum_rmse += np.square(r - pr)
    rmse = np.sqrt(sum_rmse / count)
    return rmse
