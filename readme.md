# Tensorflow speech recognition challenge

This is my solution for the [tensorflow speech recognition challenge on kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).

## Requirements
+ tensorflow v1.4
+ scipy for wave loading
+ librosa for audio manipulations
+ numpy, pandas
+ pyaudio for microphone

## Usage
The model can be trained using the *train.py*, all the parameters that can be defined can be found in the parser settings or using `python train.py --help`.\\
The file *settings1.csv* contains a definition of the parameters that will be used to run a particular experiment.\\
Once the model has been trained, it can be used to predict samples using *predict.py*, passing the previously generated checkpoints.\\
The file *mic.py* is used for real-time recognition, using the microphone and the pyAudio library.
