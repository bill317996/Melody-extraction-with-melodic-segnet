# Melody-extraction-with-melodic-segnet

The source code of "A Streamlined Encoder/Decoder Architecture for Melody Extraction"

### Dependencies

Requires following packages:

- python 3.6
- pytorch 0.4.1
- numpy
- scipy
- pysondfile

### Usage
#### predict_on_audio.py
Melody extraction on an audio file.
The output will be .txt file of time(s) and frequency(Hz).
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
  -m mode                 The mode of CFP: std and fast (default: fast)
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

### Todos

 - Add codes for training_process
 - Add codes for training_data_manage
 - Add figures for case-study

### Cleaning up codes and will upload it soon!
