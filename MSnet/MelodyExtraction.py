import torch
import numpy as np
from MSnet.cfp import cfp_process
import MSnet.model as model

def est(output, CenFreq, time_arr):
    
    CenFreq[0] = 0
    est_time = time_arr
    output = output[0,0,:,:]
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]
        
    est_arr = np.concatenate((est_time[:,None],est_freq[:,None]),axis=1)
    return est_arr


def seg(data,seg_frames_length=5120):
    frames = data.shape[-1]
    cutnum = int(frames / seg_frames_length)
    remain = frames - (cutnum*seg_frames_length)
    xlist = []
    for i in range(cutnum):
        x = data[:,:, i*seg_frames_length:(i+1)*seg_frames_length]
        xlist.append(x)
    if frames % seg_frames_length != 0:
        x = data[:,:, cutnum*seg_frames_length:]
        xlist.append(x)
    return xlist
def iseg(data, seg_frames_length=256):
    x = data[0]
    for i in range(len(data)-1):
        x = np.concatenate((x, data[i+1]), axis=-1)
    return x
def MeExt(filepath, model_type='vocal', model_path='./pretrain_model/MSnet_vocal', GPU=True, mode='std'):
    if 'std' in mode:
        data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=44100, hop=256)
    elif 'fast' in mode:
        data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=22050, hop=512)
    print('Melody extraction with Melodic Segnet ...')
    if 'vocal' in model_type:
        Net = model.MSnet_vocal()
    elif 'melody' in model_type:
        Net = model.MSnet_melody()
    else: 
        print("Error: Wrong type of model. Please assign model_type = 'vocal' or 'melody'")
        return None
    
    Net.float()
    Net.eval()
    
    if GPU:
        Net.cuda()
        Net.load_state_dict(torch.load(model_path, map_location={'cuda:2':'cuda:{}'.format(gid)}))
    else:
        Net.cpu()
        Net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    frames = data.shape[-1]
    if frames > 5120:
        seg_x = seg(data)
        seg_y=[]
        for batch_x in seg_x:
            batch_x = batch_x[np.newaxis,:]
            batch_x = torch.from_numpy(batch_x).float()
            if GPU:
                batch_x = batch_x.cuda()

            pred_y, emb = Net(batch_x)
            pred_y = pred_y.cpu().detach().numpy()
            seg_y.append(pred_y)
        pred_y = iseg(seg_y)
    else:

        batch_x = data[np.newaxis,:]
        batch_x = torch.from_numpy(batch_x).float()
        if GPU:
            batch_x = batch_x.cuda()

        pred_y, emb = Net(batch_x)
        pred_y = pred_y.cpu().detach().numpy()
        
    est_arr = est(pred_y, CenFreq, time_arr)
    return est_arr
