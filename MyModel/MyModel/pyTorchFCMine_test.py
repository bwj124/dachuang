import torch.utils.data
from torch import nn
from model import MLP

project_root = 'F:/others/'
data_root = project_root+'1_data/'

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
# net = MLP().to(device)
criteon = nn.BCEWithLogitsLoss().to(device)

data_name = '2021_04_08_18_29_13'
test_data = torch.load(data_root+'data_{}.pt'.format(data_name))
test_target = torch.load(data_root+'target_{}.pt'.format(data_name))
model_name = '2021_04_08_17_46_01'
net = torch.load('F:/others/models/model_{}.pt'.format(model_name))


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
    with open('F:/others/logs/log_test.txt', 'a') as f:
        f.write(output_str + '\n' + f'TP: {TP}\tTN: {TN}\tFP: {FP}\tFN: {FN}' + '\n')


test_model(net, test_data, test_target)

