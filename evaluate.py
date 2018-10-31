import os
import torch
import mir_eval
import csv
import argparse
import pandas as pd
import numpy as np
from MSnet.utils import getlist_ADC2004, getlist_MIREX05, getlist_mdb, getlist_mdb_vocal, melody_eval, select_vocal_track
from MSnet.MelodyExtraction import MeExt

def main(data_dir, model_type, output_dir, gpu_index, dataset='Mdb_vocal'):
    model_path = './MSnet/pretrain_model/MSnet_'+str(model_type)
    if 'ADC2004' in dataset:
        songlist = getlist_ADC2004()
    elif 'MIREX05' in dataset:
        songlist = getlist_MIREX05()
    elif 'Mdb_vocal' in dataset:
        _, _, songlist = getlist_mdb_vocal()
    elif 'Mdb_melody2' in dataset:
        _, _, songlist = getlist_mdb()
    else:
        print('Error: Wrong type of dataset, Must be ADC2004 or MIREX05 or Mdb_vocal or Mdb_melody2')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir+'/'+str(dataset)+'_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Songname', 'VR', 'VFA', 'RPA', 'RCA', 'OA'])
        avg_arr = [0,0,0,0,0]
        for songname in songlist:
            if 'Mdb_vocal' in dataset:
                filepath = data_dir+'/Audio/'+songname+'/'+songname+'_MIX.wav'
                ypath = data_dir+'/Annotations/Melody_Annotations/MELODY2/'+songname+'_MELODY2.csv'
                lpath = data_dir+'Annotations/Instrument_Activations/SOURCEID/'+songname+'_SOURCEID.lab'
                ref_arr = select_vocal_track(ypath, lpath)
            elif 'Mdb_melody2' in dataset:
                filepath = data_dir+'/Audio/'+songname+'/'+songname+'_MIX.wav'
                ypath = data_dir+'/Annotations/Melody_Annotations/MELODY2/'+songname+'_MELODY2.csv'
                ycsv = pd.read_csv(ypath, names = ["time", "freq"])
                gtt = ycsv['time'].values
                gtf = ycsv['freq'].values
                ref_arr = np.concatenate((gtt[:,None], gtf[:,None]), axis=1)
            else:
                filepath = data_dir+'/'+songname+'.wav'
                ypath = data_dir+'/'+songname+'REF.txt'
                ref_arr = np.loadtxt(ypath)
               
            if gpu_index is not None:
                with torch.cuda.device(gpu_index):
                    est_arr = MeExt(filepath, model_type=model_type, model_path=model_path, GPU=True)
            else:
                est_arr = MeExt(filepath, model_type=model_type, model_path=model_path, GPU=False)
            # np.savetxt(output_dir+'/'+songname+'.txt', est_arr)
                
            eval_arr = melody_eval(ref_arr, est_arr)
            avg_arr += eval_arr
            writer.writerow([songname, eval_arr[0], eval_arr[1], eval_arr[2], eval_arr[3], eval_arr[4]])
        avg_arr /= len(songlist)
        writer.writerow(['Avg', avg_arr[0], avg_arr[1], avg_arr[2], avg_arr[3], avg_arr[4]])
def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-dd', '--data_dir',
                    help='Path to the dataset folder (default: %(default)s',
                    type=str, default='Dataset/MedleyDB/Source/')
    p.add_argument('-t', '--model_type',
                    help='Model type: vocal or melody (default: %(default)s',
                    type=str, default='vocal')
    p.add_argument('-gpu', '--gpu_index',
                    help='Assign a gpu index for processing. It will run with cpu if None.  (default: %(default)s',
                    type=int, default=0)
    p.add_argument('-o', '--output_dir',
                    help='Path to output foler (default: %(default)s',
                    type=str, default='../output/')
    p.add_argument('-ds', '--dataset' ,
                    help='Dataset for evaluate: Must be ADC2004 or MIREX05 or Mdb_vocal or Mdb_melody2 (default: %(default)s',
                    type=str, default='Mdb_vocal')
    return p.parse_args()
if __name__ == '__main__':
    args = parser()
    main(args.data_dir, args.model_type, args.output_dir, args.gpu_index, args.dataset)