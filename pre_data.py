import scipy.io as sio
import torch
import torch.utils.data as Data
import torch.nn as nn

import numpy as np
from kernel_trick.kernel import GaussianKernel
import h5py

from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class Dataset(Data.Dataset):
    def __init__(self,x1, x2, gt):
        self.x1 = x1
        self.x2 = x2
        self.gt = gt

    def __getitem__(self, index):
        d1, d2, target = self.x1[index], self.x2[index], self.gt[index]
        return d1, d2, target

    def __len__(self):
        return len(self.x1)

def load_data(DATA_PATH):
    # pre-processing data
    loaded_data = sio.loadmat(DATA_PATH)

    data_x1 = loaded_data['x1']
    data_x2 = loaded_data['x2']
    labels = loaded_data['gt'][0]

    index_0 = [a for a, b in enumerate(labels) if b == 0]
    index_1 = [a for a, b in enumerate(labels) if b == 1]
    index_2 = [a for a, b in enumerate(labels) if b == 2]
    data_x1 = np.delete(data_x1, index_1, axis=0)
    data_x2 = np.delete(data_x2, index_1, axis=0)
    labels = np.delete(labels, index_1, axis=0)
    # sel = VarianceThreshold(threshold=(0.15 * (1 - .8)))
    # data_x1 = sel.fit_transform(data_x1)
    # data_x2 = sel.fit_transform(data_x2)
    data_x1 = SelectKBest(chi2, k=40).fit_transform(data_x1, labels)
    data_x2 = SelectKBest(chi2, k=20).fit_transform(data_x2, labels)

    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # sel.fit_transform(data)

    # lassocv = LassoCV()
    # lassocv.fit(data_x1, labels)
    # mask = lassocv.coef_ != 0
    # data_x1 = data_x1[:, mask]

    data_x1 = torch.from_numpy(scaler(data_x1)).type(torch.FloatTensor)
    data_x2 = torch.from_numpy(scaler(data_x2)).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    dataset = Dataset(data_x1, data_x2, labels)
    loader = Data.DataLoader(dataset=dataset, batch_size=data_x1.size(0), shuffle=True, num_workers=0)
    for step, (x1, x2, labels) in enumerate(loader):
        break

    return x1, x2, labels

def getNormLaplacian(W):
    D = sum(W)
    sqrtDegreeMatrix = np.diag(1.0 / (D ** 0.5))
    # D = torch.diag(D)
    # L = D - W
    k = np.dot(np.dot(sqrtDegreeMatrix, W), sqrtDegreeMatrix)
    L = np.eye(W.shape[0])-k
    np.fill_diagonal(L, 1.0)
    L = (torch.from_numpy(L)).type(torch.float32)

    return L

def hw_load(DATA_PATH):
    dataset = sio.loadmat(DATA_PATH)
    x1, x2, y = dataset['x1'], dataset['x2'], dataset['gt']
    # x1, x2, y = x1.value, x2.value, y.value
    # x1, x2, y = x1.transpose(), x2.transpose(), y.transpose()

    # tmp = np.zeros(len(y))
    # y = np.reshape(y, np.shape(tmp))
    x1, x2 = scaler(x1), scaler(x2)
    x1 = torch.from_numpy(x1).type(torch.FloatTensor)
    x2 = torch.from_numpy(x2).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.LongTensor)

    return x1, x2, y[0, :]

def scaler(matrix):
    # matrix.numpy()
    min_max_scaler = MinMaxScaler()
    matrix_minmax = min_max_scaler.fit_transform(matrix)
    # matrix_minmax = torch.from_numpy(matrix_minmax)

    return matrix_minmax

def similarity_rbf(H1):
    H1 = H1.cpu().data.numpy()
    # similarity = torch.from_numpy(cosine_similarity(H1, H2))
    kernel = GaussianKernel.GaussianKernel(sigma=1.0)
    similarity = scaler(kernel.evaluate(H1, H1))
    similarity = ((similarity + similarity.T) / 2).astype(np.float32)
    np.fill_diagonal(similarity, 1.0)
    similarity = torch.from_numpy(similarity)
    return similarity

def similarity_cos(H1, H2):
    similarity = scaler(cosine_similarity(H1, H2))
    # similarity = cosine_similarity(H1, H2)
    similarity = torch.from_numpy(similarity)

    return similarity

def get_k(ac, type):
    kernel = GaussianKernel.GaussianKernel(sigma=1.0)
    # Kycc = scaler(kernel.evaluate(ac, ac))
    Kycc = kernel.evaluate(ac, ac)
    Kycc = ((Kycc + Kycc.T) / 2).astype(np.float32)
    np.fill_diagonal(Kycc, 1.0)
    if type == 'np':
        return Kycc
    if type == 'torch':
        Kycc = torch.from_numpy(Kycc).type(torch.float32)
        return Kycc

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit / n

    return np.dot(np.dot(Q, K), Q)
    
def F_norm(A, B):
    K = np.trace(np.matmul(A, B))
    F = pow(K, 1/2)
    return F

def kernel_alig(K1, K2):
    ker_alig = (F_norm(K1, K2))/pow((F_norm(K1, K1)*F_norm(K2, K2)), 1/2)

    return ker_alig
    
def process_sim(labels, class_num):
    x = labels.size(0)
    labels = (labels/2).unsqueeze(-1)
    one_hot = torch.zeros(labels.size(0), class_num).scatter_(1, labels, 1)
    sim_gt = cosine_similarity(one_hot.numpy(), one_hot.numpy())
    sim_gt = torch.from_numpy(sim_gt).type(torch.FloatTensor)
    return sim_gt