from test.cluster import cluster
import warnings
warnings.filterwarnings('ignore')

def result(n_clusters, H, gt):
    H, gt = H.cpu().data.numpy(), gt.cpu().data.numpy()

    acc_avg, acc_std, nmi_avg, nmi_std, fscore_avg, fscore_avg, recall_avg, recall_std = cluster(n_clusters, H, gt, count=10)
    # print('clustering h      : acc = {:.4f}, nmi = {:.4f}'.format(acc_H, nmi_H))

    return acc_avg, acc_std, nmi_avg, nmi_std, fscore_avg, fscore_avg, recall_avg, recall_std