
import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

import h5py
import mir_eval
import pickle

from MSnet.model import MSnet_vocal, MSnet_melody
import argparse

class Dataset(Data.Dataset):
    def __init__(self, data_tensor, target_tensor):

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def est(output, CenFreq, time_arr):
    
    CenFreq[0] = 0
    est_time = time_arr
    output = output[0,0,:,:]
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]
        
    est_arr = np.concatenate((est_time[:,None],est_freq[:,None]),axis=1)
    return est_arr

def melody_eval(ref, est):

    ref_time = ref[:,0]
    ref_freq = ref[:,1]

    est_time = est[:,0]
    est_freq = est[:,1]

    output_eval = mir_eval.melody.evaluate(ref_time,ref_freq,est_time,est_freq)
    VR = output_eval['Voicing Recall']*100.0 
    VFA = output_eval['Voicing False Alarm']*100.0
    RPA = output_eval['Raw Pitch Accuracy']*100.0
    RCA = output_eval['Raw Chroma Accuracy']*100.0
    OA = output_eval['Overall Accuracy']*100.0
    eval_arr = np.array([VR, VFA, RPA, RCA, OA])
    return eval_arr

def pos_weight(data):
    frames = data.shape[-1]
    freq_len = data.shape[-2]
    non_vocal = np.sum(data[:,0,:]) * 1.0
    vocal = (len(data) * frames) - non_vocal
    z = np.zeros((freq_len, frames))
    z[1:,:] += (non_vocal / vocal)
    z[0,:] += vocal / non_vocal
    return torch.from_numpy(z).float().cuda()

def main(fp, model_type, gid, op, epoch_num, learn_rate, bs):
    if 'vocal' in model_type:
        Net = model.MSnet_vocal()
    elif 'melody' in model_type:
        Net = model.MSnet_melody()
    if gid is not None:
        Net.cuda()
    else:
        Net.cpu()
    Net.float()

    epoch_num = 10000
    bs = 50
    learn_rate = 0.0001
    """
    Loading training data:
        training data shape should be x: (n, 3, freq_bins, time_frames) extract from audio by cfp_process
                                      y: (n, 1, freq_bins+1, time_frames) from ground-truth
    """

    print('Loading training data ...')
    hf = h5py.File(fp+'/train.h5', 'r')
    x = hf.get('x')[:]
    y = hf.get('y')[:]
    CenFreq = hf.get('CenFreq')[:]
    time_arr = hf.get('time_arr')[:]
    hf.close()

    """
    Loading Validation data
    """
    with open(fp+'/val_x.pickle', 'rb') as file:
        x_test_list = pickle.load(file)

    with open(fp+'/val_y.pickle', 'rb') as file:
        y_test_list = pickle.load(file)

    pw = pos_weight(all_gt)

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    data_set = Dataset(data_tensor=x_tensor, target_tensor=y_tensor)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=bs, shuffle=True)

    """
    Training
    """
    
    best_epoch = 0
    best_OA = 0

    BCELoss = nn.BCEWithLogitsLoss(pos_weight=pw)
    # BCELoss = nn.BCEWithLogitsLoss()
    opt = optim.Adam(Net.parameters(), lr=learn_rate)

    for epoch in range(epoch_num):

        start_time = time.time()
        Net.train()
        train_loss = 0
            
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            pred = Net(batch_x.cuda())
            loss = BCELoss(pred, batch_gt.cuda())
            loss.backward()
            opt.step()
            train_loss += loss.item()

        Net.eval()
        avg_eval_arr = np.array([0,0,0,0,0],dtype='float64')
        for i in range(len(x_test_list)):
            x_test = x_test_list[i][None,:]
            x_test = torch.from_numpy(x_test).float()
            pred = Net(x_test.cuda())
            pred = pred.cpu().detach().numpy()
            
            y_test = y_test_list[i]

            ref_arr = y_test
            est_arr = est(pred, CenFreq, time_arr)
            eval_arr = melody_eval(est_arr, ref_arr)
            avg_eval_arr += eval_arr

        avg_eval_arr /= len(x_test_list)
        print('=========================')
        print('Epoch: ',epoch,' | train_loss: %.4f'% train_loss)
        print('Valid | VR: {:.2f}% VFA: {:.2f}% RPA: {:.2f}% RCA: {:.2f}% OA: {:.2f}%'.format(
            avg_eval_arr[0], avg_eval_arr[1], avg_eval_arr[2], avg_eval_arr[3], avg_eval_arr[4]))
        
        if avg_eval_arr[-1] > best_OA:
            best_OA = avg_eval_arr[-1]
            best_epoch = epoch
            torch.save(Net.state_dict(), op+'/model_'+model_type) 
        
        print('Best Epoch: ', best_epoch, ' | Best OA: %.2f'%best_OA)
        print('Time: ', int(time.time()-start_time), '(s)')

def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-fp', '--filepath',
                    help='Path to input training data (h5py file) and validation data (pickle file) (default: %(default)s)',
                    type=str, default='./train/data/')
    p.add_argument('-t', '--model_type',
                    help='Model type: vocal or melody (default: %(default)s)',
                    type=str, default='vocal')
    p.add_argument('-gpu', '--gpu_index',
                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s)',
                    type=int, default=None)
    p.add_argument('-o', '--output_dir',
                    help='Path to output folder (default: %(default)s)',
                    type=str, default='./train/model/')
    p.add_argument('-ep', '--epoch_num', 
                    help='the number of epoch (default: %(default)s)',
                    type=int, default=100)
    p.add_argument('-lr', '--learn_rate', 
                    help='the number of learn rate (default: %(default)s)',
                    type=float, default=0.0001)
    p.add_argument('-bs', '--batch_size', 
                    help='The number of batch size (default: %(default)s)',
                    type=int, default=50)
    return p.parse_args()
if __name__ == '__main__':

    args = parser()
    if args.gpu_index is not None:
        with torch.cuda.device(args.gpu_index):
            train(args.filepath, args.model_type, args.gpu_index, args.output_dir, args.epoch_num, args.learn_rate, args.batch_size)
    else: 
        train(args.filepath, args.model_type, args.gpu_index, args.output_dir, args.epoch_num, args.learn_rate, args.batch_size)

