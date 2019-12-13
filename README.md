# 790-Project
This project consists of three parts: STGAN, MTGAN and SGAN

## SGAN
SGAN is written in tensorflow, but the model structure has some references to the original Theano implementation: 
https://github.com/zalandoresearch/spatial_gan <br>
Except the OS modules, image processing functions and the flags set-ups, most codes are written by myself.

To use SGAN, simpy run python run.py <br>
You must have a directory called dataset that has all training samples in it. <br>
To change parameters, use --   For example, python run.py --spatial_size=7<br>
To generate images from pretrained models, use python run.py --training=False --model_dir="<The directory where your models are in>"
  
Here are some good samples:

![1](SGAN/samples/cracked.jpg)
![2](SGAN/samples/fibrous.jpg)
![3](SGAN/samples/marble3.jpg)
![4](SGAN/samples/stars.jpg)
![5](SGAN/samples/wood.jpg)
