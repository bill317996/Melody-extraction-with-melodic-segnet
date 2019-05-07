# Melody-extraction-with-melodic-segnet

The source code of "A Streamlined Encoder/Decoder Architecture for Melody Extraction"
- arxiv: https://arxiv.org/abs/1810.12947

### Dependencies

Requires following packages:

- python 3.6
- pytorch 0.4.1
- numpy
- scipy
- pysoundfile
- pandas


### Usage
#### predict_on_audio.py
Melody extraction on an audio file.
The output will be .txt file of time(sec) and frequency(Hz).
```
usage: predict_on_audio.py [-h] [-fp FILEPATH] [-t MODEL_TYPE]
                           [-gpu GPU_INDEX] [-o OUTPUT_DIR] [-e EVALUATE]

optional arguments:
  -h
  -fp filepath            Path to input audio(.wav) (default: train01.wav)
  -t model_type           Model type: vocal or melody (default: vocal)
  -gpu gpu_index          Assign a gpu index for processing.
                          It will run with cpu if None. (default: 0)
  -o output_dir           Path to output folder (default: ./output/)
  -e evaluate             Path to ground-truth (default: None)
  -m mode                 The mode of CFP: std and fast (default: std)
                          fast mode: use sr=22050 and hop=512 (faster)
                          std mode : use sr=native_sample_rate and hop=256 (more accurate)
```
#### evaluate.py
Evaluate our result on three dataset: ADC2004, MIREX05, MedleyDB.
The output will be .csv file of evaluation metrics (mir_eval).
```
usage: evaluate.py [-h] [-dd DATA_DIR] [-t MODEL_TYPE] [-gpu GPU_INDEX]
                   [-o OUTPUT_DIR] [-ds DATASET]
optional arguments:
  -h
  -dd data_dir          Path to the dataset folder (default:
                        Dataset/MedleyDB/Source/)
  -t model_type         Model type: vocal or melody (default: vocal)
  -gpu gpu_index        Assign a gpu index for processing.
                        It will run with cpu if None. (default: 0)
  -o output_dir         Path to output foler (default: ./output/)
  -ds dataset           Dataset for evaluate (default: Mdb_vocal)
                        Must be ADC2004 or MIREX05 or Mdb_vocal or Mdb_melody2 
```

#### training.py
Please prepare the h5py file for training data first from your audios by cfp_process 
```
usage: training.py [-h] [-fp FILEPATH] [-t MODEL_TYPE] [-gpu GPU_INDEX]
                   [-o OUTPUT_DIR] [-ep EPOCH_NUM] [-lr LEARN_RATE]
                   [-bs BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -fp FILEPATH, --filepath FILEPATH
                        Path to input training data (h5py file) and validation
                        data (pickle file) (default: ./train/data/)
  -t MODEL_TYPE, --model_type MODEL_TYPE
                        Model type: vocal or melody (default: vocal)
  -gpu GPU_INDEX, --gpu_index GPU_INDEX
                        Assign a gpu index for processing. It will run with
                        cpu if None. (default: None)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output folder (default: ./train/model/)
  -ep EPOCH_NUM, --epoch_num EPOCH_NUM
                        the number of epoch (default: 100)
  -lr LEARN_RATE, --learn_rate LEARN_RATE
                        the number of learn rate (default: 0.0001)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        The number of batch size (default: 50)
```
