import numpy as np
from MSnet.cfp import cfp_process
from MSnet.utils import getlist_mdb, getlist_mdb_vocal, select_vocal_track, csv2ref
import argparse
import h5py
import pickle

def seq2map(seq, CenFreq):
    CenFreq[0] = 0
    gtmap = np.zeros((len(CenFreq),len(seq)))
    for i in range(len(seq)):
        for j in range(len(CenFreq)):
            if seq[i] < 0.1:
                gtmap[0,i] = 1
                break
            elif CenFreq[j] > seq[i]:
                gtmap[j,i] = 1
                break
    return gtmap

def batchize(data, gt, xlist, ylist, size=430):
    if data.shape[-1] != gt.shape[-1]:
        new_length = min(data.shape[-1], gt.shape[-1])
        data = data[:,:,:new_length]
        gt = gt[:,:new_length]
    num = int(gt.shape[-1]/size)
    if gt.shape[-1]%size != 0:
        num += 1
    for i in range(num):
        if (i+1)*size > gt.shape[-1]:
            batch_x = np.zeros((data.shape[0],data.shape[1],size))
            batch_y = np.zeros((gt.shape[0],size))
            
            tmp_x = data[:,:,i*size:]
            tmp_y = gt[:,i*size:]
            
            batch_x[:,:,:tmp_x.shape[-1]] += tmp_x
            batch_y[:,:tmp_y.shape[-1]] += tmp_y
            xlist.append(batch_x)
            ylist.append(batch_y)
            break
        else:
            batch_x = data[:,:,i*size:(i+1)*size]
            batch_y = gt[:,i*size:(i+1)*size]
            xlist.append(batch_x)
            ylist.append(batch_y)
            
    return xlist, ylist

def batchize_val(data, size=430):
    xlist = []
    num = int(data.shape[-1]/size)
    if data.shape[-1]%size != 0:
        num += 1
    for i in range(num):
        if (i+1)*size > data.shape[-1]:
            batch_x = np.zeros((data.shape[0],data.shape[1],size))
            
            tmp_x = data[:,:,i*size:]
            
            batch_x[:,:,:tmp_x.shape[-1]] += tmp_x
            xlist.append(batch_x)
            break
        else:
            batch_x = data[:,:,i*size:(i+1)*size]
            xlist.append(batch_x)
            
    return np.array(xlist)

def main(data_folder, model_type, output_folder):  
    if 'vocal' in model_type:
        train_songlist, val_songlist, test_songlist = getlist_mdb_vocal()
        xlist = []
        ylist = []
        for songname in train_list:
            filepath = data_folder + '/Audio/' + songname + '/' + songname + '_MIX.wav'
            data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=44100, hop=256)
            ypath = data_folder + '/Annotations/Melody_Annotations/MELODY2/' + songname + '_MELODY2.csv'
            lpath = data_folder + '/Annotations/Instrument_Activations/SOURCEID/' + songname + '_SOURCEID.lab'
            ref_arr = select_vocal_track(ypath, lpath)
            gt_map = seq2map(ref_arr[:,1], CenFreq)

            xlist, ylist = batchize(data, gt_map, xlist, ylist, size=430)

        xlist = np.array(xlist)
        ylist = np.array(ylist)
        hf = h5py.File('./data/train_vocal.h5', 'w')
        hf.create_dataset('x', data=xlist)
        hf.create_dataset('y', data=ylist) 
        hf.close()
        
        xlist = []
        ylist = []
        for songname in val_songlist:
            filepath = data_folder + '/Audio/' + songname + '/' + songname + '_MIX.wav'
            data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=44100, hop=256)
            data = batchize_val(data)
            ypath = data_folder + '/Annotations/Melody_Annotations/MELODY2/' + songname + '_MELODY2.csv'
            lpath = data_folder + '/Annotations/Instrument_Activations/SOURCEID/' + songname + '_SOURCEID.lab'
            ref_arr = select_vocal_track(ypath, lpath)
            
            xlist.append(data)
            ylist.append(ref_arr)
            
        with open(output_folder+'/val_x_vocal.pickle', 'wb') as fp:
            pickle.dump(xlist, fp)

        with open(output_folder+'/val_y_vocal.pickle', 'wb') as fp:
            pickle.dump(ylist, fp)

    elif 'melody' in model_type:
        train_songlist, val_songlist, test_songlist = getlist_mdb()
        xlist = []
        ylist = []
        for songname in train_list:
            filepath = data_folder + '/Audio/' + songname + '/' + songname + '_MIX.wav'
            data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=44100, hop=256)
            ypath = data_folder + '/Annotations/Melody_Annotations/MELODY2/' + songname + '_MELODY2.csv'
            ref_arr = csv2ref(ypath)
            gt_map = seq2map(ref_arr[:,1], CenFreq)

            xlist, ylist = batchize(data, gt_map, xlist, ylist, size=430)

        xlist = np.array(xlist)
        ylist = np.array(ylist)
        hf = h5py.File(output_folder+'/train.h5', 'w')
        hf.create_dataset('x', data=xlist)
        hf.create_dataset('y', data=ylist) 
        hf.close()
        
        xlist = []
        ylist = []
        for songname in val_songlist:
            filepath = data_folder + '/Audio/' + songname + '/' + songname + '_MIX.wav'
            data, CenFreq, time_arr = cfp_process(filepath, model_type=model_type, sr=44100, hop=256)
            data = batchize_val(data)
            ypath = data_folder + '/Annotations/Melody_Annotations/MELODY2/' + songname + '_MELODY2.csv'
            ref_arr = csv2ref(ypath)
            
            xlist.append(data)
            ylist.append(ref_arr)
            
        with open(output_folder+'/val_x.pickle', 'wb') as fp:
            pickle.dump(xlist, fp)

        with open(output_folder+'/val_y.pickle', 'wb') as fp:
            pickle.dump(ylist, fp)

def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-df', '--data_folder',
                    help='Path to the dataset folder (default: %(default)s',
                    type=str, default='./data/MedleyDB/Source/')
    p.add_argument('-t', '--model_type',
                    help='Model type: vocal or melody (default: %(default)s',
                    type=str, default='vocal')
    p.add_argument('-o', '--output_folder',
                    help='Path to output foler (default: %(default)s',
                    type=str, default='./data/')
    return p.parse_args()
if __name__ == '__main__':
    args = parser()
    main(args.data_folder, args.model_type, args.output_folder)
