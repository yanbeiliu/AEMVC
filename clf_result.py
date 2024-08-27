from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
def clf(H, labels):
    kf = KFold(n_splits=10, shuffle=True)
    cla_acc, clf_auc = [], []
    for i in range(50):
        for train, test in kf.split(H):
            break
        xtrain, xtest, ytrain, ytest = H[train], H[test], labels.numpy()[train]-1, labels.numpy()[test]-1
        # clf = KNeighborsClassifier(n_neighbors=20, weights='distance')
        clf = SVC(gamma=1, probability=True)
        # clf = KNeighborsClassifier(n_neighbors=50)
        # xtrain, xtest, ytrain, ytest = x1[0:300, :].numpy(), x1[300:360, :].numpy(), labels[0:300].numpy(), labels[300:360].numpy()
        clf.fit(xtrain, ytrain)
        # x = clf.predict_proba(xtest)
        # index_prob = [clf.predict_proba(xtest).T[ytest[i], i] for i in range(ytest.shape[0])]
        # fpr, tpr, thresholds = metrics.roc_curve(ytest, index_prob)
        # test_auc = auc(fpr, tpr)
        # clf_auc.append(test_auc)
        # print(index)
        pre_acc = clf.score(xtest, ytest)
        # print(pre_acc)
        cla_acc.append(pre_acc)
    cla_acc, cla_std = np.average(cla_acc), np.std(cla_acc)
    # clf_auc = np.average(clf_auc)
    return cla_acc, cla_std
