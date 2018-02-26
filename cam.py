import mxnet as mx
import sys
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
import cv2
import os


def forward(x, net, ctx, con):
    layers = net.features[:-2]
    x = nd.array(x, ctx=ctx)
    for layer in layers:
        x = layer(x)
    cams = con(x)
    x = nn.GlobalAvgPool2D()(x)
    x = nn.Flatten()(x)
    predictions = net.output(x)
    return predictions, cams

def Cam(network, image, model_path, ctx, data_shape, class_names, thresh,num_class=14):
    net = getattr(models,network)(classes=num_class)

    with net.name_scope(): 
        net.output = nn.Dense(num_class,activation="sigmoid")
    net.output.initialize(init.Xavier())

    params_features = os.path.join(model_path, 'densenet_cam_f.params')
    params_output = os.path.join(model_path, 'densenet_cam_o.params')
    net.features.load_params(params_features, ctx=ctx)
    net.output.load_params(params_output,ctx=mx.cpu(0))
    net.collect_params().reset_ctx(ctx)
    net.hybridize()

    params = net.output.collect_params()
    class_weights = params[list(params.keys())[0]]

    c=nn.Conv2D(channels=num_class,kernel_size=1)
    c.initialize(ctx=ctx)
    test = nd.random.normal(shape=(32,1024, 7, 7), ctx=ctx)
    c(test)
    c.weight.set_data(class_weights.data().reshape((num_class,1024,1,1)))
     
    n = len(image)
    X = np.zeros((n,3,data_shape,data_shape),dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(n):
        img = cv2.imread(image[i])
        X[i] = ((cv2.resize(img, (data_shape, data_shape))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
    
    predictions, cams = forward(X[np.arange(n)],net,ctx,c)
    predictions = predictions.asnumpy()
    cams = cams.asnumpy()

    for i in range(n):
        img = cv2.imread(image[i])
        if (predictions[i]>thresh).any():
            for j in range(num_class):
                if predictions[i,j]>thresh:
                    cam = cams[i][j]
                    cam -= cam.min()
                    cam /= cam.max()
                    cam = cv2.resize((cam * 255).astype(np.uint8), (img.shape[1], img.shape[0]))
                    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                    out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
                    cv2.imshow('Image:%s pred:%s'%(image[i],class_names[j]),out)
        else:
            print('No finding in:%s'%image[i])
    cv2.waitKey (0)  
    cv2.destroyAllWindows()  