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

```
usage: predict_on_audio.py [-h] [-fp FILEPATH] [-t MODEL_TYPE]
                           [-gpu GPU_INDEX] [-o OUTPUT_DIR] [-e EVALUATE]

optional arguments:
  -h
  -fp filepath            Path to input audio (default: train01.wav)
  -t model_type           Model type: vocal or melody (default: vocal)
  -gpu gpu_index          Assign a gpu index for processing.
                          It will run with cpu if None. (default: 0)
  -o output_dir           Path to output folder (default: ./output/)
  -e evaluate             Path of ground truth (default: None)
```

### Todos

 - Add codes for training_process
 - Add codes for training_data_manage

### Cleaning up codes and will upload it soon!
