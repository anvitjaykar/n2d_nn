import argparse
import os
import random as rn
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils.linear_assignment_ import linear_assignment
from time import time

rn.seed(0)
tf.set_random_seed(0)
np.random.seed(0)


np.set_printoptions(threshold=sys.maxsize)

def cluster_manifold_in_embedding(hl, y, label_names=None):
    # find manifold on autoencoded embedding
        md = 20
        hle = umap.UMAP(
            random_state=0,
            metric=5,
            n_components=10,
            n_neighbors=20,
            min_dist=md).fit_transform(hl)

        km = KMeans(
            init='k-means++',
            n_clusters=10,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)

    y_pred = np.asarray(y_pred)
    # y_pred = y_pred.reshape(len(y_pred), )
    y = np.asarray(y)
    # y = y.reshape(len(y), )
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print("mnist | UMAP on autoencoded embedding with Kmeans - N2D")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    return y_pred, acc, nmi, ari


def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)


if __name__ == "__main__":

    optimizer = 'adam'
    from datasets import load_mnist, load_mnist_test

    label_names = None
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'mnist-test':
        x, y = load_mnist_test()

    shape = [x.shape[-1], 500, 500, 2000, args.n_clusters]
    autoencoder = autoencoder(shape)

    hidden = autoencoder.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden)

    pretrain_time = time()

    # Pretrain autoencoders before clustering
    autoencoder.compile(loss='mse', optimizer=optimizer)
    autoencoder.fit(
            x,
            x,
            batch_size=args.batch_size,
            epochs=args.pretrain_epochs,
            verbose=0)
    pretrain_time = time() - pretrain_time
    autoencoder.save_weights('weights/' +
                                 args.dataset +
                                 "-" +
                                 str(args.pretrain_epochs) +
                                 '-ae_weights.h5')
    print("Time to train the autoencoder: " + str(pretrain_time))


    hl = encoder.predict(x)
    clusters, t_acc, t_nmi, t_ari = cluster_manifold_in_embedding(
        hl, y, label_names)