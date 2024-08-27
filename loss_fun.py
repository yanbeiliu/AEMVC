import torch
import torch.nn as nn

import pre_data
import numpy as np

from scipy.stats import pearsonr

def corre_loss(H1, H2):
    c = torch.cosine_similarity(H1, H2, dim=1)
    nc = -torch.norm(c, p=1)
    return nc
    # c = pearsonr(H1, H2)
    #
    # return -torch.norm(c, 1)

def graph_reg(L1, z, D, lamb):
    k = torch.matmul(z.t(), D)
    st = torch.matmul(k, z)
    m_loss = nn.MSELoss()
    loss = lamb[0] * torch.trace(torch.matmul(torch.matmul(z.t(), L1), z)) +\
           lamb[1] * m_loss(st, torch.eye(st.shape[0]).cuda())
    return loss

def lap_loss(z, K):
    Z = torch.matmul(z, z.t())
    loss = 1/2 * torch.trace(torch.matmul(K.t(), Z))
    return loss

def ce_loss(logits, gt):
    loss = nn.CrossEntropyLoss()
    ce = loss(logits, gt)
    return ce

def hsic(z, k2):
    # k1 = pre_data.similarity_cos(z.data.numpy(), z.data.numpy())
    # k1 = pre_data.get_k(z, 'torch')
    k1 = torch.matmul(z, z.t())
    # # k1 = t orch.from_numpy(pre_data.scaler(torch.matmul(z, z.t()).data.numpy()))
    n = k1.size(0)
    e = torch.ones(n, n)
    H = (torch.eye(n) - (1/n) * e).cuda()
    hsic = -((n-1)**(-2))*(torch.trace(torch.matmul(torch.matmul(torch.matmul(k1, H), k2), H)))
    # hsic = torch.log(hsic)
    # k12 = torch.matmul(k1, k2)
    # H = torch.trace(k12)/n**2 + torch.mean(k1)*torch.mean(k2) - 2*torch.mean(k12)/n
    # hsic = H*n**2/(n-1)**2
    # hsic = -torch.log(hsic)

    return hsic

