import os.path
import numpy as np
import matplotlib.pyplot as plt


# Question 1
def pca(X, k=100):
    mean_vec = X.mean(axis=0)
    mean_X = X - mean_vec

    cov_X = np.cov(mean_X, rowvar=0)
    eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_X))
    eig_val_index = np.argsort(eig_vals)
    eig_val_index = eig_val_index[:-(k + 1): -1]

    top_k_eig_vecs = eig_vecs[:, eig_val_index]
    reduced_X = mean_X * top_k_eig_vecs
    recons_X = (reduced_X * top_k_eig_vecs.T) + mean_X

    return mean_vec, eig_vals, top_k_eig_vecs, recons_X


def plot(X, top_eig_vec=20, dst_prefix=""):
    mean_vec, eig_vals, top_k_eig_vecs, recons_X = pca(X)

    # plot the mean image
    display(mean_vec.reshape(1, 784)[0, :], dst_prefix + "/mean_img.png")

    # plot the top eigen-vector
    for i in range(0, top_eig_vec):
        vec = top_k_eig_vecs.T[i, :]
        display(vec.astype(float), dst_prefix + "/top_eigenvecs_" + str(i) + ".png")

    # plot the top 100 eigenvalues
    plot_eigenvalues(eig_vals, 100, dst_prefix + "/eigenvalues.png")


def plot_eigenvalues(eig_vals, top_rank=100, dstpath=""):
    rank = range(0, top_rank)
    index = np.arange(len(rank))
    plt.bar(index, list(eig_vals)[0:top_rank])
    plt.xlabel('Rank')
    plt.ylabel('EigenValue')
    plt.title('Top ' + str(top_rank) + ' EigenValues')
    if dstpath != "":
        plt.savefig(dstpath)
    plt.show()


def display(Xrow, dstpath=""):
    ''' Display a digit by first reshaping it from the row-vector into the image.  '''
    plt.imshow(np.reshape(Xrow, (28, 28)))
    plt.gray()
    if dstpath != "":
        plt.savefig(dstpath)
    plt.show()


def load_data(digit=0, num=200):
    ''' 
    Loads all of the images into a data-array (for digits 0 through 5). 

    The training data has 5000 images per digit and the testing data has 200, 
    but loading that many images from the disk may take a while.  So, you can 
    just use a subset of them, say 200 for training (otherwise it will take a 
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and 
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.
    
    '''
    X = np.zeros((num, 784), dtype=np.uint8)  # 784=28*28
    print '\nReading digit %d' % digit,
    for i in xrange(num):
        if not i % 100: print '.',
        pth = os.path.join('mnist-subset/train%d' % digit, '%05d.pgm' % i)
        with open(pth, 'rb') as infile:
            header = infile.readline()
            header2 = infile.readline()
            header3 = infile.readline()
            image = np.fromfile(infile, dtype=np.uint8).reshape(1, 784)
        X[i, :] = image
    print '\n'
    return X


Q2_X = load_data(0, 5000)
# Question 2
plot(Q2_X, 20, "./q2")

# Question 3
Q3_X = load_data(2, 5000)
plot(Q3_X, 20, "./q3")

Q4_X = load_data(1, 5000)
Q4_X = np.concatenate((Q2_X, Q3_X, Q4_X))
plot(Q4_X, 20, "./q4")
