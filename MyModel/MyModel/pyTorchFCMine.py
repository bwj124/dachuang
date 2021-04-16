# import torch
import random
import torch.utils.data
from torch import optim, nn
import numpy as np
import pandas as pd
from model import MLP
# import torch.nn.functional as F

batch_size = 1000
# batch_size = 1
learning_rate = 0.001
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
#             nn.Linear(202, 2)
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#
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

name = '2021_04_02_17_39_45'
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


for epoch in range(epochs):

    # for batch_idx, (data, target) in enumerate(train_loader):
    for i in range(0, train_data.shape[0], batch_size):
        if i + batch_size > train_data.shape[0]:
            end = train_data.shape[0]
        else:
            end = i + batch_size
        batch_data = train_data[i:end]
        batch_data = batch_data.view(-1, 300)
        batch_data = batch_data.to(device)

        logits = net(batch_data)
        # computed_values.append(logits)
        # if logits.item() < 0 or logits.item() > 1:
        #     logits.item() = logits.item()/logits.item()+1
        loss = criteon(logits, train_target[i:end])
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

        labels = torch.argmax(train_target[i:end], 1)
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
        if (epoch+1) % 100 == 0:
            print('Train Epoch : {}\tbatch: {}\tLoss: {:.6f}\taccuracy: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(epoch+1, i//batch_size, loss, accuracy, precision, recall, f1))


    #
    #     if batch_idx % 100 == 0:
    #         print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx*len(data), len(train_loader.dataset),
    #             100.*batch_idx/len(train_loader), loss.item()
    #         ))


    # test_loss = 0
    # correct = 0
    # for data, target in test_loader:
    # for i in range(test_data.shape[0]):
    # data = data.view(-1, 28*28)
    if epoch == epochs - 1:
        test_data, test_target = test_data.to(device), test_target.to(device)
        logits = net(test_data)
        test_loss = criteon(logits, test_target).item()

        labels = torch.argmax(test_target, 1)
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
        print('Test : Loss: {:.6f}\taccuracy: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
            test_loss, accuracy, precision, recall, f1
        ))

    # pred = logits.data.max(1)[1]
    # correct += pred.eq(target.data).sum()

    # test_loss /= test_data.shape[0]
    # print('\nTest set : Averge loss: {:.4f}, Accurancy: {}/{}({:.3f}%)'.format(
    #     test_loss, correct, test_data.shape[0],
    #     100.*correct/test_data.shape[0]
    #     ))

    #
