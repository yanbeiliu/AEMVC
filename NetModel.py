import torch.nn as nn
import torch

from loss_fun import corre_loss
import loss_fun


"""
Net framework: two Mlps

"""

class Net(nn.Module):
    def __init__(self, input1_dim, half_dim, L1, lamb):
        super(Net, self).__init__()
        self.lamb = lamb
        self.L1 = L1
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input1_dim),
            nn.Linear(input1_dim, input1_dim*2),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input1_dim*2),
            nn.Linear(input1_dim*2, input1_dim*2),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input1_dim*2),
            nn.Linear(input1_dim*2, half_dim),
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(half_dim),
            nn.Linear(half_dim, input1_dim*2),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input1_dim*2),
            nn.Linear(input1_dim*2, input1_dim*2),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input1_dim*2),
            nn.Linear(input1_dim*2, input1_dim),
        )

    def forward(self, x1):
        z = self.encoder(x1)
        x1_ge = self.decoder(z)

        return x1_ge, z

    def total_loss(self, x1, x1_ge, z, D, k2):
        re_loss = nn.MSELoss()
        loss1 = re_loss(x1, x1_ge)
        loss2 = loss_fun.graph_reg(self.L1, z, D, self.lamb)
        loss3 = loss_fun.hsic(z, k2)
        loss4 = loss_fun.lap_loss(z, k2)
        # print(loss1)
        # print(loss2)
        # print(self.lamb[2]*loss3)
        # print(self.lamb[3]*loss4)
        total_loss = loss1 + loss2 + self.lamb[2]*loss3 + self.lamb[3]*loss4

        return total_loss

class CLF(nn.Module):
    def __init__(self, input_dim, num_class):
        super(CLF, self).__init__()
        self.clf = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim*2, input_dim*2),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Tanh(),
            nn.BatchNorm1d(input_dim*2),
            nn.Linear(input_dim*2, num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        logits = self.clf(X)

        return logits
