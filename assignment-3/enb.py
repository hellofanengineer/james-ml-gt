
__author__ = 'james'

from sklearn.feature_extraction import DictVectorizer
from numpy import *
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pylab as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GMM
from sklearn.decomposition import *
from scipy import stats
from sklearn.utils import extmath
import matplotlib.ticker as ticker
#from sklearn import preprocessing
#from itertools import combinations
#from collections import defaultdict


def read_csv(file_path):
    rows = []
    target = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 2:
                break

            r = {}
            for i in range(0, len(parts)):
                p = parts[i].strip()
                if i == 0:
                    r["compactness"] = float(p)
                elif i == 1:
                    r['surface-area'] = float(p)
                elif i == 2:
                    r['wall-area'] = float(p)
                elif i == 3:
                    r['roof-area'] = float(p)
                elif i == 4:
                    r['overall-height'] = float(p)
                elif i == 5:
                    r['orientation'] = float(p)
                elif i == 6:
                    r['glazing-area'] = float(p)
                elif i == 7:
                    r['glazing-area-dist'] = float(p)
                elif i == 8:
                    target.append(float(p))

            rows.append(r)

    return rows, target


def plot_chart(y, y_label, title, x_label):
    pl.plot(range(2, len(y) + 2), y, lw=2)
    pl.title(title)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.show()


def reduce_and_learn_nn(reducer, train_data, train_target, test_data, test_target):
    # train
    reducer.fit(train_data, train_target)
    # execute the reduction
    r_train_X = reducer.transform(train_data)
    # apply the reduction to the test data also
    r_test_X = reducer.transform(test_data)

    err, dur = learn_nn(r_train_X, train_target, r_test_X, test_target)
    return err, dur


def learn_nn(train_data, train_target, test_data, test_target):
    #print(train_target)
    # X features and 1 target label
    dimension = len(train_data[0])
    ds = SupervisedDataSet(dimension, 1)
    for i in range(len(train_data)):
        ds.addSample(train_data[i], train_target[i])
    #print("train=%d,test=%d" % (len(train_data), len(test_data)))

    # build a FeedForwardNetwork with 1 hidden layer
    fnn = buildNetwork(dimension, 5, 1)

    # create a back propagation trainer
    trainer = BackpropTrainer(fnn, dataset=ds)
    x = []
    train_err_res = []
    test_err_res = []
    t1 = time.clock()
    lowest = -1
    for i in range(50):
        #print("Starting fit at %s..." % time.strftime("%H:%M:%S", time.localtime()))
        start = time.clock()
        train_err = trainer.train()
        finish = time.clock()
        dur = finish-start
        #print("train complete in %02d:%02d. err:%s" % (int(int(dur) / 60), int(dur) % 60, str(train_err)))

        y = zeros(len(test_data))
        for j in range(len(test_data)):
            y[j] = fnn.activate(test_data[j])

        dur2 = time.clock()-dur
        #print("test complete in %02d:%02d" % (int(int(dur2) / 60), int(dur2) % 60))

        test_err = mean_squared_error(y, test_target)

        x.append(i)
        train_err_res.append(train_err)
        test_err_res.append(test_err)

        if lowest == -1 or test_err < lowest:
            lowest = test_err

        #print "epoch: %4d" % trainer.totalepochs, \
        #    "  train error: %5.4f" % train_err, \
        #    "  test error: %5.4f" % test_err

    t2 = time.clock()
    print('lowest error: %5.4f. total time = %.2f' % (lowest, (t2-t1)))
    return lowest, t2-t1


def parallel_coordinates(data_sets, style=None, title=''):
    print('len pc data=%d' % len(data_sets))
    dims = len(data_sets[0])
    x = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)
    fig.suptitle(title)

    if style is None:
        style = ['r-']*len(data_sets)

    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) /
                min_max_range[dimension][2]
                for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi])
        ax.set_xlim([x[i], x[i+1]])

    # Set the x axis ticks
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in xrange(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)


    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in xrange(ticks)]
    axx.set_yticklabels(labels)

    # Stack the subplots
    plt.subplots_adjust(wspace=0)

    return plt


def plot_clusters(data, clusters, title):
    cmap = "bgrcmykw"
    cluster_colors = []
    for cluster in clusters:
        cluster_colors.append(cmap[cluster])

    parallel_coordinates(data, style=cluster_colors, title=title).show()


def clusterK(train_data, test_data, max):
    best = -1
    best_k = 0
    scores = []
    best_rs = -1
    for rs in range(20):
        for k in range(2, max+1):
            km = KMeans(n_clusters=k, random_state=rs)
            score, dur = score_clustering(km, train_data, test_data)
            if score > best:
                best = score
                best_k = k
                best_rs = rs
                print('local best k=%d, rs=%d, score=%.3f' % (best_k, best_rs, best))

    print('KMeans k=%d, rs=%d, score=%.3f' % (best_k, best_rs, best))

    # plot the clusters for best k
    km = KMeans(n_clusters=best_k, random_state=best_rs).fit(train_data)
    test_clusters = km.predict(test_data)
    plot_clusters(test_data, test_clusters, 'KMeans Clusters k=%d' % best_k)

    # for best random state, iterate through k to return scores
    for k in range(2, max+1):
        km = KMeans(n_clusters=k, random_state=best_rs)
        score, dur = score_clustering(km, train_data, test_data)
        scores.append(score)
        if k == best_k:
            print('KM duration: %d' % dur)

    return best_rs, scores

def clusterEM(train_data, test_data, max):
    best = -1
    best_k = 0
    scores = []
    best_rs = -1
    for rs in range(20):
        for k in range(2, max+1):
            em = GMM(n_components=k, random_state=rs)
            score, dur = score_clustering(em, train_data, test_data)
            if score > best:
                best = score
                best_k = k
                best_rs = rs
                print('local best k=%d, rs=%d, score=%.3f' % (best_k, best_rs, best))

    print('EM k=%d, rs=%d, score=%.3f' % (best_k, best_rs, best))

    em = GMM(n_components=best_k, random_state=best_rs).fit(train_data)
    clusters = em.predict(test_data)
    plot_clusters(test_data, clusters, 'EM Clusters c=%d' % best_k)

    for k in range(2, max+1):
        em = GMM(n_components=k, random_state=rs)
        score, dur = score_clustering(em, train_data, test_data)
        scores.append(score)
        if k == best_k:
            print('EM duration: %d' % dur)

    print('EM k=%d, score=%.3f' % (best_k, best))
    return best_rs, scores

'''
def dunn_index(X, labels, cluster_centers):
    """Computes the Dunn index of the clusters. Implemented according to the
    description here: http://en.wikipedia.org/wiki/Cluster_analysis"""

    # Calculate the minimum inter-cluster distance
    min_inter = min([
        linalg.norm(a, b) \
        for a, b in combinations(cluster_centers, 2)])

    # Group points by label
    by_label = defaultdict(list)
    for l, x in zip(labels, X): by_label[l].append(x)

    # Calculate the maximum intra-cluster point distances
    max_intra = max([
        max([linalg.norm(a, b) \
            for a, b in combinations(xs, 2)]) \
        for xs in by_label.values() if len(xs) > 1])

    if max_intra == 0:
        return 0

    return min_inter / max_intra
'''

def score_clustering(km, data, test_data):
    t0 = time.time()
    # fit the training data
    km.fit(data)
    # put test data into clusters
    clusters = km.predict(test_data)
    duration = time.time() - t0
    # score the clustering
    v = metrics.silhouette_score(test_data, clusters, metric='euclidean')
    return v, duration


def cluster_and_learn_nn(train_data, train_target, test_data, test_target,):
    # get cluster assignments for training and test data
    # 2 was the best k per earlier experiments
    km = KMeans(n_clusters=2, random_state=1).fit(train_data)
    train_clusters = km.predict(train_data)
    test_clusters = km.predict(test_data)
    # add the cluster assignment as a feature
    train_with_cluster = np.concatenate((train_data, train_clusters.reshape(len(train_clusters), 1)), axis=1)
    test_with_cluster = np.concatenate((test_data, test_clusters.reshape(len(test_clusters), 1)), axis=1)

    print('KMeans cluster NN')
    learn_nn(train_with_cluster, train_target, test_with_cluster, test_target)

    # repeat with EM
    # 4 = best c per earlier experiments
    em = GMM(n_components=4, random_state=1)
    em.fit(train_data)
    train_clusters = em.predict(train_data)
    test_clusters = em.predict(test_data)
    # add the cluster assignment as a feature
    train_with_cluster = np.concatenate((train_data, train_clusters.reshape(len(train_clusters), 1)), axis=1)
    test_with_cluster = np.concatenate((test_data, test_clusters.reshape(len(test_clusters), 1)), axis=1)

    print('EM cluster NN')
    learn_nn(train_with_cluster, train_target, test_with_cluster, test_target)


def plot_eigenvalues(alg, train_data, title):
    alg.fit(train_data)
    eig_vals = alg.explained_variance_ratio_
    print(eig_vals)
    pl.figure()
    pl.plot(range(1, len(eig_vals) + 1), eig_vals, lw=2)
    pl.ylabel('eigenvalue')
    pl.title('%s eigenvalues' % title)
    pl.show()


def reduce_and_cluster(reduceFn, train_data, test_data, max_features, reduced_features, max_k, title, rs=1):
    best = -1
    best_k = 0
    best_time = 0
    scores = []

    # track reconstruction error
    recon_err = []
    for c in range(1, max_features+1):
        reducer = reduceFn(c, rs).fit(train_data)
        test_reduced = reducer.transform(test_data)
        if not hasattr(reducer, 'inverse_transform'):
            break
        # calculate the error as euclidean norm
        err = linalg.norm(test_data - reducer.inverse_transform(test_reduced))
        print('%s k=%d,reconstruction err=%.4f' % (title, c, err))
        recon_err.append(err)

    if len(recon_err) > 0:
        #plot recon err
        pl.plot(range(1, max_features+1), recon_err, lw=2)
        pl.ylabel('Norm')
        pl.xlabel('Components')
        pl.title('%s Reconstruction Error' % title)
        pl.show()

    # create dimensionality reducer and fit training data
    reducer = reduceFn(reduced_features, rs).fit(train_data)
    # execute the reduction
    fit = reducer.transform(train_data)
    # apply the reduction to the test data also
    test_fit = reducer.transform(test_data)

    # find the best k
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, random_state=1)
        score, t = score_clustering(km, fit, test_fit)
        scores.append(score)
        #print('KM score %d:%.3f in %d' % (k, score, t))
        if score > best:
            best = score
            best_k = k
            best_time = t

    print('%s: best k=%d,score=%.3f,t=%d' % (title, best_k, best, best_time))
    plot_chart(scores, 'Silhouette Score', title + ' - Reduced KMeans', 'k')

    km = KMeans(n_clusters=best_k, random_state=1).fit(fit)
    test_k_clusters = km.predict(test_fit)
    plot_clusters(test_fit, test_k_clusters, title + ' Reduced KMeans Clusters')

    best = -1
    scores = []
    # run with EM
    for c in range(2, max_k):
        em = GMM(n_components=c, random_state=1)
        score, t = score_clustering(em, fit, test_fit)
        scores.append(score)
        #print('EM score %d:%.3f in %d' % (c, score, t))
        if score > best:
            best = score
            best_k = c
            best_time = t

    print('%s: best c=%d,score=%.3f,t=%d' % (title, best_k, best, best_time))
    plot_chart(scores, 'Silhouette Score', title + ' - Reduced EM', 'Components')

    em = GMM(n_components=best_k, random_state=1).fit(fit)
    test_em_clusters = em.predict(test_fit)
    plot_clusters(test_fit, test_em_clusters, title + ' - Reduced EM Clusters')



def find_best_rp(train_data, test_data, start_components, max_features):
    # find random proj with lowest reconstruction error
    best_c = 0
    #scores = []
    #best_k = 0
    #best = 0
    best_rs = 0
    best_r = 20000

    # go to max - 1 since it doesn't make sense to randomly project to the same dimension
    r = range(start_components, max_features)
    print(r)
    # center data for reconstruction
    scalar = StandardScaler(with_mean=True, with_std=False)
    centered = scalar.fit_transform(test_data)
    for c in r:
        print('C=%d' % c)
        for rs in range(1, 501):
            rp = GaussianRandomProjection(n_components=c, random_state=rs).fit(train_data)
            fit = rp.transform(centered)
            recon = extmath.safe_sparse_dot(fit, rp.components_) + scalar.mean_
            err = linalg.norm(test_data - recon)
            if err < best_r:
                best_r = err
                best_c = c
                best_rs = rs
    print('best reconstruction error=%.4f' % best_r)
    print('>>best rs=%d,c=%d' % (best_rs, best_c))

    # for the best, track the variation
    v_max = 0
    errsum = 0
    for rs in range(1, 501):
        rp = GaussianRandomProjection(n_components=c, random_state=rs).fit(train_data)
        fit = rp.transform(centered)
        recon = extmath.safe_sparse_dot(fit, rp.components_) + scalar.mean_
        err = linalg.norm(test_data - recon)
        errsum += err
        if err > v_max:
            v_max = err

    print('RP max:%.3f, avg:%.3f' % (v_max, errsum/500))

    return best_c, best_rs


def build_pca(num_components, rs):
    return PCA(n_components=num_components)


def build_ica(num_components, rs):
    return FastICA(n_components=num_components, max_iter=18000, random_state=rs)


def build_svd(num_components, rs):
    return TruncatedSVD(n_components=num_components, algorithm='arpack', random_state=rs)


def build_rp(num_components, rs):
    return GaussianRandomProjection(n_components=num_components, random_state=rs)


def cluster(data_array, target_array, n_components, perform_nn, max_k, max_features):
    # normalize numeric values for better performance per docs
    data_array = StandardScaler().fit_transform(data_array, target_array)

    target2 = None
    test_target = None

    if target_array is not None:
        data2, target2 = shuffle(data_array, target_array, random_state=1)
    else:
        data2 = shuffle(data_array, random_state=1)

    # split training and testing data 70/30
    offset = int(0.7*len(data2))
    train_data = data2[:offset]
    test_data = data2[offset:]

    if target_array is not None:
        train_target = target2[:offset]
        test_target = target2[offset:]

    best_k_rs, scores = clusterK(train_data, test_data, max_k)
    plot_chart(scores, 'Silhouette Score', 'K Means', 'k')

    best_em_rs, scores = clusterEM(train_data, test_data, max_k)
    plot_chart(scores, 'Silhouette Score', 'EM', 'Components')

    plot_eigenvalues(PCA(), train_data, 'PCA')
    t0 = int(round(time.time() * 1000))
    PCA().fit_transform(train_data)
    t1 = int(round(time.time() * 1000))
    pca_time = t1-t0
    print('PCA time: %d ms' % pca_time)

    reduce_and_cluster(build_pca, train_data, test_data, max_features, n_components, max_k, 'PCA')

    # calculate and plot kurtosis to determine any components that can be dropped
    t0 = int(round(time.time() * 1000))
    ica = FastICA(max_iter=18000, random_state=1)
    s = ica.fit_transform(train_data)
    t1 = int(round(time.time() * 1000))
    ica_time = t1-t0
    print('ICA time: %d ms' % ica_time)
    z_score, p_score = stats.kurtosistest(s)
    pl.plot(z_score, 'ro')
    pl.title('Kurtosis Z-scores')
    pl.ylabel('Z-score')
    pl.xlabel('Component Index')
    pl.show()
    reduce_and_cluster(build_ica, train_data, test_data, max_features, n_components, max_k, 'ICA')

    rp_c, rp_rs = find_best_rp(train_data, test_data, 2, max_features)

    reduce_and_cluster(build_rp, train_data, test_data, max_features, rp_c, max_k, 'RP', rp_rs)

    reduce_and_cluster(build_svd, train_data, test_data, max_features-1, n_components, max_k, 'SVD')

    if perform_nn:
        errs = []
        durs = []
        err, dur = reduce_and_learn_nn(PCA(n_components=n_components), train_data, train_target, test_data, test_target)
        errs.append(err)
        durs.append(dur)
        # ICA had 4 components as the best
        err, dur = reduce_and_learn_nn(FastICA(max_iter=18000, random_state=1, n_components=4), train_data,
                                       train_target, test_data, test_target)
        errs.append(err)
        durs.append(dur)
        # num components = 7 for RP
        err, dur = reduce_and_learn_nn(GaussianRandomProjection(n_components=rp_c, random_state=rp_rs), train_data,
                                       train_target, test_data, test_target)
        errs.append(err)
        durs.append(dur)
        err, dur = reduce_and_learn_nn(TruncatedSVD(n_components=n_components, algorithm='arpack', random_state=1), train_data,
                                       train_target, test_data, test_target)
        errs.append(err)
        durs.append(dur)
        ind = np.arange(len(errs))
        bar_width = 0.3
        fig, ax = plt.subplots()
        ax.bar(ind, errs, bar_width, color='b')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Error After Dimension Reduction')
        ax.set_xticks(ind+.15)
        ax.set_xticklabels(('PCA', 'ICA', 'RP', 'SVD'))
        pl.show()

        cluster_and_learn_nn(train_data, train_target, test_data, test_target)


# dataset 1
data_array, data_target = read_csv("./enb2012.csv")
vec = DictVectorizer()
data_array = vec.fit_transform(data_array).toarray()

# 6 = best dimension reduction per experiments, 10 = #features+2, 8 = # features
cluster(data_array, data_target, 6, True, 10, 8)

# dataset 2
w_data = np.genfromtxt('./wholesale.csv', skip_header=1, delimiter=',')

# 7 = best dimension reduction per experiments, 10 = #features+2, 8 = # features
cluster(w_data, None, 7, False, 10, 8)