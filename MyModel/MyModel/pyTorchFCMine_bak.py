# import torch
import os
import random
import time

import torch.utils.data
from torch import optim, nn
import numpy as np
import pandas as pd
from time import sleep

from model import MLP
# import torch.nn.functional as F

batch_size = 30
# batch_size = 1
learning_rate = 0.0001
epochs = 20000
project_root = 'F:/others/'
data_root = project_root+'data/'

# class MLP(nn.Module):
#
#     def __init__(self):
#         super(MLP, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(300, 202),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(202, 121),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(121, 2)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


device = torch.device('cpu')
net = MLP().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criteon = nn.BCEWithLogitsLoss().to(device)


# astnn_data = torch.load('candidate_encode.pt')
# ggnn_data = np.load('final_node.npy')
# ggnn_data = torch.from_numpy(ggnn_data)
# data = torch.zeros(ggnn_data.shape[0], astnn_data.shape[1]+ggnn_data.shape[1])
# target = torch.Tensor([[0, 1]])
# for i in range(ggnn_data.shape[0]-1):
#     a = random.randrange(2)
#     temp = torch.Tensor([[a, 1-a]])
#     target = torch.cat((target, temp), dim=1)
# # target = target.repeat(ggnn_data.shape[0], 1)
# target = target.reshape((ggnn_data.shape[0], 2))
# for i in range(ggnn_data.shape[0]):
#     data[i][:astnn_data.shape[1]] = astnn_data
#     data[i][astnn_data.shape[1]:] = ggnn_data[i]
# print('all data has contacted')
# global index
# index = 0

name = '2021_04_08_16_29_42'
data = torch.load(data_root+'data_{}.pt'.format(name))
target = torch.load(data_root+'target_{}.pt'.format(name))


def split_train_test(datasets, targets, test_ratio):
    np.random.seed(random.randrange(50))
    shuffled_indices = np.random.permutation(len(datasets))
    test_set_size = int(len(datasets) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return datasets.iloc[train_indices].values.tolist(), \
           datasets.iloc[test_indices].values.tolist(), \
           targets.iloc[train_indices].values.tolist(), \
           targets.iloc[test_indices].values.tolist()


train_data, test_data, train_target, test_target = split_train_test(pd.DataFrame(data.tolist()), pd.DataFrame(target.tolist()), 0.3)
train_data, test_data, train_target, test_target = torch.Tensor(train_data), torch.Tensor(test_data), torch.Tensor(train_target), torch.Tensor(test_target)

time_temp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def train_model(epoch, net, train_data, train_target):
# for i in range(0, train_data.shape[0], batch_size):
    # if i + batch_size > train_data.shape[0]:
    #     end = train_data.shape[0]
    # else:
    #     end = i + batch_size
    # batch_data = train_data[i:end]
    # batch_data = batch_data.view(-1, 300)
    # batch_data = batch_data.to(device)
    #
    # logits = net(batch_data)
    # # computed_values.append(logits)
    # # if logits.item() < 0 or logits.item() > 1:
    # #     logits.item() = logits.item()/logits.item()+1
    # loss = criteon(logits, train_target[i:end])
    batch_data = train_data.to(device)
    logits = net(batch_data)
    loss = criteon(logits, train_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # loss = torch.tensor([loss])
    # if i == 0:
    #     computed_values = logits
    #     take_loss = loss
    # else:
    #     computed_values = torch.cat([computed_values, logits], dim=0)
    #     take_loss = torch.cat([take_loss, loss], dim=0)
    #

# computed_values = np.array(computed_values, dtype=np.float32)
# # computed_values = torch.Tensor(computed_values)
# computed_values = torch.from_numpy(computed_values)
# take_loss = np.array(take_loss, dtype=np.float32)
# # take_loss = torch.Tensor(take_loss)
# take_loss = torch.from_numpy(take_loss)

    # labels = torch.argmax(train_target[i:end], 1)
    labels = torch.argmax(train_target, 1)

    # prediction = torch.argmax(computed_values, 1)
    prediction = torch.argmax(logits, 1)
    accuracy = torch.mean(torch.eq(prediction, labels).float())
    # take_loss = torch.mean(take_loss)

    TP = torch.sum(prediction * labels).float()
    TN = torch.sum((1 - prediction) * (1 - labels)).float()
    FP = torch.sum(prediction * (1 - labels)).float()
    FN = torch.sum((1 - prediction) * labels).float()
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # if i // 60:
    i = 0
    output_str = 'Train Epoch : {}\tLoss: {:.6f}\taccuracy: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(epoch, loss, accuracy, precision, recall, f1)
    if epoch % 100 == 0:
        print(output_str)
        logFile = 'F:/others/logs/log_{}.txt'.format(time_temp)
        # if not os.path.exists(logFile):
        #     os.mknod(logFile)
        with open(logFile, 'a') as f:
            f.write(output_str + '\n' + f'TP: {TP}\tTN: {TN}\tFP: {FP}\tFN: {FN}' + '\n')


def test_model(net, test_data, test_target):
    test_data, test_target = test_data.to(device), test_target.to(device)
    logits = net(test_data)
    test_loss = criteon(logits, test_target).item()

    labels = torch.argmax(test_target, 1)
    prediction = torch.argmax(logits, 1)
    accuracy = torch.mean(torch.eq(prediction, labels).float())

    TP = torch.sum(prediction * labels).float()
    TN = torch.sum((1 - prediction) * (1 - labels)).float()
    FP = torch.sum(prediction * (1 - labels)).float()
    FN = torch.sum((1 - prediction) * labels).float()
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    output_str = 'Test: \tLoss: {:.6f}\taccuracy: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
        test_loss, accuracy, precision, recall, f1
    )
    print(output_str)
    with open('F:/others/logs/log_{}.txt'.format(time_temp), 'a') as f:
        f.write(output_str + '\n' + f'TP: {TP}\t TN: {TN}\t FP: {FP}\t FN: {FN}' + '\n')


for epoch in range(epochs):
    train_model(epoch+1, net, train_data, train_target)

sleep(1)
test_model(net, test_data, test_target)

torch.save(net, 'F:/others/models/model_{}.pt'.format(time_temp))