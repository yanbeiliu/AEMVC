import torch
import torch.functional as F
import torch.nn as nn

import NetModel
import pre_data
import complete as com
import kernel_trick.KPCA as KPCA
from kernel_trick.KCCA import KernelCCA
from test.misc import evaluateKMeans, visualizeData
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from clf_result import clf

import matplotlib.pyplot as plt
import numpy as np
import time
import progressbar
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DATA_PATH = './data/ADNI_shuffle.mat'
EPOCH = 100
# lamb = [1, 100, 1, 1e-5]
# lamb = [0.1, 0, 0.1, 1e-5]
lamb = [0.01, 0, 0.1, 1e-5]
# lamb = [0, 0, 0, 0]
learning_rate = 0.001
H_dim = 20
topk = 20
M = 153
num_class = 2
acc, inter = [], []
loss_inter = []
clf_acc = []
p = progressbar.ProgressBar(EPOCH)

kcca = KernelCCA(tau=1)
x1, x2, labels = pre_data.load_data(DATA_PATH)

# H = torch.cat((x1, x2), dim=1)
# visualizeData(x2.numpy(), labels.numpy(), num_class, 'Original')

# ac = x2[0:M, :]
# am = np.random.random((x1.size(0)-ac.size(0), x2.size(1)))
# am = (torch.from_numpy(am)).type(torch.float32)
# random_x2 = torch.cat((ac, am), 0)
# # H = torch.cat((x1, random_x2), dim=1)
# visualizeData(random_x2.numpy(), labels.numpy(), num_class, 'Random')

target_kernel = pre_data.process_sim(labels, num_class)
train_x1 = x1.cuda()
train_x2 = x2.cuda()
train_labels = labels.cuda()
ac = x2[0:M, :].numpy()
print('ground truth x1 clustering：')
x1_result = evaluateKMeans(train_x1.cpu().numpy(), train_labels.cpu().numpy(), num_class)
print(x1_result[0])
print('ground truth x2 clustering：')
x2_result = evaluateKMeans(train_x2.cpu().numpy(), train_labels.cpu().numpy(), num_class)
print(x2_result[0])

K2cc = (pre_data.get_k(ac, 'torch')).cuda()
K2 = com.complete(K2cc, train_x1)
# K1 = pre_data.get_k(train_x1.cpu().numpy(), 'np')
# alpha, beta, lmbdas = kcca.learnModel(K1, K2)
# p1, p2 = kcca.project(K1, K2, k=40)
# P = np.c_[p1, p2]
#####################################################################################################
# tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30)
# vis_data = tsne.fit_transform(com_x2)
# fig = plt.figure()
# # for a, b, c in zip(vis_data[0:500, 0], vis_data[0:500, 1], ground_labels[0:500]):
# #     plt.text(a, b+0.001, '%d' % c, ha='center', va='bottom', fontsize=9)
# fig = plt.gcf()  # get current figure
# ax1 = fig.add_subplot(111)
# miss_precent = str((N/(N+M))*100) + "%"
# ax1.set_title(miss_precent)
# plt.scatter(vis_data[:, 0], vis_data[:, 1], c=train_labels, marker='o', s=10, cmap=plt.cm.get_cmap("jet", 10))
# plt.clim(-0.5, 6.5)
# plt.show()
#########################################################################################################################

# w1 = pre_data.similarity_rbf(train_x1)
w1 = pre_data.get_k(train_x1.cpu().numpy(), 'torch')
# w1 = pre_data.process_sim(labels, num_class)
L1 = pre_data.getNormLaplacian(w1.numpy()).cuda()
D = torch.diag(sum(w1)).cuda()
net1 = NetModel.Net(train_x1.size(1), H_dim, L1, lamb)
net1.cuda()
net_clf = NetModel.CLF(topk*2, num_class)
net_clf.cuda()
optim1 = torch.optim.Adam(net1.parameters(), lr=learning_rate)
optim2 = torch.optim.Adam(net_clf.parameters(), lr=0.001)
loss_clf = nn.CrossEntropyLoss()


# p.start()
for epoch in range(EPOCH):
    net1.train()
    # tr_x1, train_gt, K2cc = train_x1.cuda(), train_labels.cuda(), K2cc.cuda()
    x1_ge, z = net1(train_x1)
    h = (torch.from_numpy(pre_data.scaler(z.cpu().data.numpy()))).cuda()
    K2 = com.complete(K2cc, h)
    loss = net1.total_loss(train_x1, x1_ge, z, D, K2)
    optim1.zero_grad()
    loss.backward()
    optim1.step()
    #print(loss)
    # p.update(epoch)

    if epoch % 1 == 0:
        net1.eval()
        inter.append(epoch)
        loss_inter.append(loss)
        # x1_ge, z = net1(train_x1)
        # tsne = TSNE(n_components=3, init='pca', random_state=0, perplexity=30)
        # vis_data = tsne.fit_transform(z.data.numpy())

        z_new = pre_data.scaler(z.cpu().data.numpy())
        # z_new = z.cpu().data.numpy()
        print('z')
        print(evaluateKMeans(z_new, train_labels.cpu().numpy(), num_class))
        K1 = pre_data.get_k(z_new, 'np')  # k1-type-numpy
        K2 = com.complete(K2cc, torch.tensor(z_new).cuda()).cpu().numpy()
        kernel_alig = pre_data.kernel_alig(K2, target_kernel)
        print(kernel_alig)
        # alpha, beta, lmbdas = kcca.learnModel(K1, K2)
        # p1, p2 = kcca.project(K1, K2, k=topk)
        # H = np.c_[p1, p2]
        K = (0.5*K1 + 0.5*K2)
        H = KPCA.kpca(K, 40)
        # if epoch % 20 == 0:
            # visualizeData(H, labels.numpy(), num_class, 'AEMVC'+str(epoch))
        #print('x2 completed clustering acc：')
        # H = pre_data.scaler(H)
        H_result = evaluateKMeans(H, train_labels.cpu().numpy(), num_class)
        acc.append(H_result[0])
        print(H_result)
        # print(H_result[0])

        clf_acc, acc_std = clf(H, train_labels.cpu())
        print(str(clf_acc) + '/' + str(acc_std))
        # plt.cla()
        # # for a, b, c in zip(vis_data[0:500, 0], vis_data[0:500, 1], ground_labels[0:500]):
        # #     plt.text(a, b+0.001, '%d' % c, ha='center', va='bottom', fontsize=9)
        #   # get current figure
        # # plt.scatter(z[:, 0].detach().numpy(), z[:, 1].detach().numpy(), c=cr[0:N], marker='o', s=10, cmap=plt.cm.get_cmap("jet", 10))
        # plt.scatter(vis_data[:, 0], vis_data[:, 1], c=cr[0:N], marker='o', s=10, cmap=plt.cm.get_cmap("jet", 10))
        # plt.clim(-0.5, 6.5)
        # plt.pause(0.1)

# p.finish()

# htrain, htest, ytrain, ytest = train_test_split(H, train_labels.cpu().numpy(), test_size=0.1, random_state=1)
# # htrain, htest, ytrain, ytest = train_test_split(H, train_labels.cpu().numpy(), test_size=0.3)
# htrain, htest, ytrain = torch.from_numpy(htrain).type(torch.FloatTensor).cuda(), torch.from_numpy(htest).type(torch.FloatTensor).cuda(), \
#                                        torch.from_numpy(ytrain).type(torch.LongTensor).cuda()
# for ep in range(200):
#     net_clf.train()
#     logits = net_clf(htrain)
#     l = loss_clf(logits, ytrain)
#     optim2.zero_grad()
#     l.backward()
#     optim2.step()
#     if ep % 100 == 0:
#         net_clf.eval()
#         logits = net_clf(htest)
#         pred_y = torch.max(logits, 1)[1].cpu().numpy()
#         accuracy = accuracy_score(ytest, pred_y)
#         print('classificaton acc:')
#         print(accuracy)

k = time.strftime('%m-%d-%H-%M', time.localtime(time.time())) + '.png'
fig = plt.figure(figsize=(10, 6))
plt.rcParams['figure.dpi'] = 200

plt.tick_params(labelsize=15)
plt.plot(inter, loss_inter,
         linestyle='--', linewidth=4, color='green',
         markersize=6, markerfacecolor='brown')
# plt.hlines(x1_result[0], inter[0], inter[-1], color='red', label='x1')
# plt.hlines(x2_result[0], inter[0], inter[-1], color='skyblue', label='x2')
# plt.legend(loc='lower right')
# plt.title(lamb)
plt.xlabel('Interation', fontsize=20)
plt.ylabel('Value of objective function', fontsize=20)
plt.savefig(k)

