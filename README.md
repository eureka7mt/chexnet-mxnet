# ChexNet-Mxnet
Still under development.This is a re-implementation of [CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/),which is a deep learning algorithm that can detect and localize 14 kinds of diseases from chest X-ray images.

<img width="450" height="450" src="https://stanfordmlgroup.github.io/projects/chexnet/img/chest-cam.png" alt="CheXNet from Stanford ML Group"/>

## CAM
It's a weakly supervised localization.Get more information from [this arxiv page](https://arxiv.org/pdf/1512.04150.pdf)

## Prerequisites
1. Python 3.5
2. [Mxnet](https://mxnet.apache.org/)
3. Numpy
4. Pandas

## Preparations
1. Clone this repository.
2. Download images of ChestX-ray14 from this [released page](https://nihcc.app.box.com/v/ChestXray-NIHCC) and decompress them to the directory [images](./images).
3. Modify the file 'Data_Entry_2017.csv' to a file like [this](./data/Data_Entry.csv).

## Try the demo

```
python demo.py
# available options
python demo.py -h
```

## Train the model

```
python train.py --gpus 0,1,2
# see advanced arguments for training
python train.py -h
```
