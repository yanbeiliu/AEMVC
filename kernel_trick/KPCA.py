from numpy import *


def kpca(K, topN):
    M = K.shape[0]
    M_M = mat(ones((M, M))) / M
    cenK = K - M_M * K - K * M_M + M_M * K * M_M
    eigVals, eigVects = linalg.eig(cenK)
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[-1: -(topN + 1): -1]
    redEigVects = eigVects[:, eigValInd]
    for i in range(topN):
        redEigVects[:, i] = redEigVects[:, i] / sqrt(eigVals[eigValInd[i]])
    needMat = cenK * redEigVects
    return needMat
