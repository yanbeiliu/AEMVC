from sklearn.cluster import KMeans
from . import metrics
import numpy as np

def cluster(n_clusters, features, labels, count):
    pred, acc, nmi, recall, fscore = np.zeros(count), np.zeros(count), np.zeros(count), np.zeros(count), np.zeros(count)
    for i in range(count):
        km = KMeans(n_clusters=n_clusters)
        pred = km.fit_predict(features)
        gt = np.reshape(labels, np.shape(pred))
        if np.min(gt) == 1:
            gt -= 1
        acc[i] = metrics.acc(gt, pred)
        nmi[i] = metrics.nmi(gt, pred)
        recall[i], fscore[i] = metrics.pre(gt, pred)
    acc_avg, acc_std = acc.mean(), acc.std()
    nmi_avg, nmi_std = nmi.mean(), nmi.std()
    recall_avg, recall_std = recall.mean(), recall.std()
    fscore_avg, fscore_std = fscore.mean(), fscore.std()
    return acc_avg, acc_std, nmi_avg, nmi_std, \
           fscore_avg, fscore_avg, recall_avg, recall_std

#
# def get_avg_acc(y_true, y_pred, count):
#     acc_array = np.zeros(count)
#     for i in range(count):
#         acc_array[i] = metrics.acc(y_true, y_pred)
#     acc_avg = acc_array.mean()
#     acc_std = acc_array.std()
#     return acc_avg, acc_std
#
#
# def get_avg_nmi(y_true, y_pred, count):
#     nmi_array = np.zeros(count)
#     for i in range(count):
#         nmi_array[i] = metrics.nmi(y_true, y_pred)
#     nmi_avg = nmi_array.mean()
#     nmi_std = nmi_array.std()
#     return nmi_avg, nmi_std
