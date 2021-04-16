import numpy as np
import torch
import os
import time

project_root = 'F:/others/'
ggnn_root = project_root+'ggnn_vector/'
astnn_root = project_root+'astnn_vector/'
combine_root = project_root+'combine_vector/'
save_root = project_root+'data/'


def contact_data():
    for _, _, files in os.walk(ggnn_root):
        # ggnn_files = [file.split('.')[1] + file.split('.')[2] for file in files]
        # astnn_files = []
        for _, _, f in os.walk(astnn_root):
            astnn_files = f
            for file in astnn_files:
                allname = file.split('-candidate_encode')[0]
                project = allname.split('.')[0]
                name = allname.split('.')[1]
                line = allname.split('.')[2]
                label = allname.split('.')[3]
                # label = file.split('-candidate_encode')[]
                for ggnn_f in files:
                    if name in ggnn_f and line in ggnn_f and label in ggnn_f and project in ggnn_f:
                        ggnn_data = np.load(ggnn_root + ggnn_f)
                        astnn_data = torch.load(astnn_root+file)
                        ggnn_data = torch.from_numpy(ggnn_data)
                        ggnn_data = ggnn_data.reshape((1, 100))
                        combine_data = torch.cat((ggnn_data, astnn_data), dim=1)
                        torch.save(combine_data, combine_root + ggnn_f.replace('.npy', '.pt'))
                        print(ggnn_f.replace('.npy', '.pt'), 'contacted')
                    else:
                        continue

    print('all data has contacted')


def generate_data():
    # generate target
    count_cursor = 0
    for _, _, files in os.walk(combine_root):
        for file in files:
            if 'truepositive' in file:
                target_temp = torch.Tensor([[0, 1]])
            else:
                target_temp = torch.Tensor([[1, 0]])
            data_temp = torch.load(combine_root + file)
            if count_cursor != 0:
                target = torch.cat((target, target_temp), dim=0)
                data = torch.cat((data, data_temp), dim=0)
            else:
                target = target_temp
                data = data_temp
            count_cursor += 1
    time_temp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    torch.save(data, save_root+'data_'+time_temp+'.pt')
    torch.save(target, save_root + 'target_' + time_temp + '.pt')
    print('all target has generated')
    print('datasets has generated')


contact_data()
# generate_data()

