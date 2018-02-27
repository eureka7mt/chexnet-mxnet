import mxnet as mx
import sys
import importlib
import re
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from loss import wSigmoidBinaryCrossEntropyLoss
import numpy as np
import pandas as pd
import cv2
import os

sigmoid_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
wsigmoid_cross_entropy = wSigmoidBinaryCrossEntropyLoss(from_sigmoid=True)


def get_optimizer_params(optimizer=None, learning_rate=None, momentum=None,
                         weight_decay=None, ctx=None):
    if optimizer.lower() == 'rmsprop':
        opt = 'rmsprop'
        print('you chose RMSProp, decreasing lr by a factor of 10')
        optimizer_params = {'learning_rate': learning_rate / 10.0,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'sgd':
        opt = 'sgd'
        optimizer_params = {'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'adadelta':
        opt = 'adadelta'
        optimizer_params = {}
    elif optimizer.lower() == 'adam':
        opt = 'adam'
        optimizer_params = {'learning_rate': learning_rate,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    return opt, optimizer_params

def compute_aucs(output,label):
    aurocs = []
    row = output.shape[0]
    column = output.shape[1]
    label_np = label.asnumpy()
    output_np = output.asnumpy()
    for i in range(column):
        if (label_np[:,i] == np.zeros((row,))).all():
            label_np[0,i] = 1-label_np[0,i]
            output_np[0,i] = 1-output_np[0,i]
            aurocs.append(roc_auc_score(label_np[:,i], output_np[:,i]))
        elif (label_np[:,i] == np.ones((row,))).all():
            label_np[0,i] = 1-label_np[0,i]
            output_np[0,i] = 1-output_np[0,i]
            aurocs.append(roc_auc_score(label_np[:,i], output_np[:,i]))
        else:
            aurocs.append(roc_auc_score(label_np[:,i], output_np[:,i]))
    return aurocs

def evaluate(net,data_iter,ctx):
    loss,n = 0., 0.
    n = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        loss += nd.mean(sigmoid_cross_entropy(output,label)).asscalar()
    return loss/n

def AUC(net,data_iter,n_classes,ctx):
    AUC = np.zeros((n_classes,))
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        auc = compute_aucs(output,label)
        auc_np = np.array(auc)
        AUC = np.row_stack((AUC,auc_np))
    m = float(AUC.shape[0]-1)
    AUCS = AUC.sum(axis=0)/m
    return  AUCS

def evaluate_resp(net, data_iter, weight, ctx):
    loss, acc, n= 0., 0., 0.
    n = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc_list=compute_acc(output,label)
        acc_list_avg=np.array(acc_list).mean()
        acc+=acc_list_avg
        loss += nd.mean(wsigmoid_cross_entropy(output, label, weight)).asscalar()
    return loss/n, acc/n

def compute_acc(output,label):
    acc = []
    row = output.shape[0]
    label_np=label.asnumpy()
    output_np=output.asnumpy()
    for i in range(row):
        if round(output_np[i,0]) == label_np[i,0]:
            acc.append(1.)
        else:
            acc.append(0.)
    
    return acc

def train_net(network, train_csv, num_classes, batch_size,
              data_shape, ctx, epochs, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              class_names=None,optimizer='sgd'):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    network : str
        name for the network structure
    train_csv : str
        .csv file path for training
    num_classes : int
        number of object classes, not including background
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    epochs : int
        epochs of training
    optimizer : str
        usage of different optimizers, other then default sgd
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    """

    # load data
    df = pd.read_csv(train_csv)
    n = len(df)
    X = np.zeros((n,3,data_shape,data_shape),dtype=np.float32)
    Y = np.zeros((n,num_classes), dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # if
    for i, dfv in enumerate(df.values):
        img = cv2.imread('./images/%s'%dfv[0])
        X[i] = ((cv2.resize(img, (data_shape, data_shape))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
        for j in range(num_classes):
            Y[i,j] = dfv[j+2]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=8)

    # fine-tune net
    pretrained_net = getattr(models, network)(pretrained=True)
    net =  getattr(models, network)(classes=num_classes)

    with net.name_scope(): 
        net.features = pretrained_net.features
        net.output = nn.Dense(num_classes,activation="sigmoid")
    net.output.initialize(init.Xavier())


    # init
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    best_auc_avg = 0

    # optimizer
    opt, opt_params = get_optimizer_params(optimizer=optimizer, learning_rate=learning_rate, momentum=momentum,
                                           weight_decay=weight_decay, ctx=ctx)

    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train,Y_train), batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_valid,Y_valid), batch_size)
    print('Running on', ctx)

    trainer = gluon.Trainer(net.collect_params(), opt,opt_params)
    for epoch in range(epochs):
        train_loss = 0.
        steps = len(train_data)
        if len(lr_refactor_step) > 0 :
            if epoch == lr_refactor_step[0]:
                trainer.set_learning_rate(trainer.learning_rate*lr_refactor_ratio)
                del lr_refactor_step[0]

        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx) 
            label_list = gluon.utils.split_and_load(label, ctx)

            with autograd.record():
                losses = [loss(net(x),y) for x,y in zip(data_list,label_list)]
            for l in losses:
                l.backward()
            
            lmean = [l.mean().asscalar() for l in losses]
            train_loss += sum(lmean)/len(lmean)
            trainer.step(batch_size)

        val_loss = evaluate(net, test_data, ctx[0])
        val_aucs = AUC(net, test_data, num_classes, ctx[0])
        val_aucs_avg = val_aucs.mean()

        print("Epoch %d. loss: %.4f, val_loss %.4f" % (
            epoch, train_loss/steps, val_loss))
        print("The average AUROC is %.3f%%" %(val_aucs_avg))

        if val_aucs_avg >= best_auc_avg:
            best_auc_avg = val_aucs_avg
            net.features.save_params('./model/densenet_cam_f_Epoch%d.params'%epoch)
            net.output.save_params('./model/densenet_cam_f_Epoch%d.params'%epoch)
            for i in range(num_classes):
                print('The AUROC of {} is {}'.format(class_names[i], val_aucs[i]))

def train_net_resp(network, train_csv, num_classes, batch_size,
                   data_shape, ctx, epochs, learning_rate,
                   momentum, weight_decay, lr_refactor_step, lr_refactor_ratio, identifier,
                   class_names=None,optimizer='sgd'):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    network : str
        name for the network structure
    train_csv : str
        .csv file path for training
    num_classes : int
        number of object classes, not including background
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    epochs : int
        epochs of training
    optimizer : str
        usage of different optimizers, other then default sgd
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    identifier : int
        identifier(number) of the object of class to classify
    """
    # load data
    df = pd.read_csv(train_csv)
    n = len(df)
    X = np.zeros((n, 3, data_shape, data_shape), dtype=np.float32)
    Y = np.zeros((n, 1), dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # if
    for i, dfv in enumerate(df.values):
        img = cv2.imread('./images/%s'%dfv[0])
        X[i] = ((cv2.resize(img, (data_shape, data_shape))[:,:,::-1] / 255.0 - mean) / std).transpose((2, 0, 1))
        Y[i,0] = dfv[identifier+2]

    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=8)
    w_train = 1.-np.sum(Y_train)/len(Y_train)
    w_val=1.-np.sum(Y_valid)/len(Y_valid)

    # fine-tune net
    pretrained_net = getattr(models,network)(pretrained=True)
    net =  getattr(models,network)(classes=1)

    with net.name_scope(): 
        net.features = pretrained_net.features
        net.output = nn.Dense(1,activation="sigmoid")
    net.output.initialize(init.Xavier())


    # init
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = wSigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    best_auc_avg = 0
    best_acc = 0

    # optimizer
    opt, opt_params = get_optimizer_params(optimizer=optimizer, learning_rate=learning_rate, momentum=momentum,
                                           weight_decay=weight_decay, ctx=ctx)

    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_train,Y_train), batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X_valid,Y_valid), batch_size)
    print('Running on', ctx)

    trainer = gluon.Trainer(net.collect_params(), opt, opt_params)
    for epoch in range(epochs):
        train_loss = 0.
        steps = len(train_data)
        if len(lr_refactor_step) > 0 :
            if epoch == lr_refactor_step[0]:
                trainer.set_learning_rate(trainer.learning_rate*lr_refactor_ratio)
                del lr_refactor_step[0]

        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx) 
            label_list = gluon.utils.split_and_load(label, ctx)

            with autograd.record():
                losses = [loss(net(x), y, w_train) for x,y in zip(data_list,label_list)]
            for l in losses:
                l.backward()
            
            lmean = [l.mean().asscalar() for l in losses]
            train_loss += sum(lmean)/len(lmean)
            trainer.step(batch_size)

        val_loss, val_acc = evaluate_resp(net, test_data, w_val, ctx[0])
        val_aucs = AUC(net, test_data, 1, ctx[0])
        val_aucs_avg = val_aucs.mean()

        print("Epoch %d. loss: %.4f, val_loss %.4f, val_acc %.2f%%" % (
            epoch, train_loss/steps, val_loss, val_acc*100))
        print('The AUROC of {} is {}'.format(class_names[identifier], val_aucs_avg))

        if val_aucs_avg >= best_auc_avg:
            best_auc_avg = val_aucs_avg
            net.features.save_params('./model/%s_f_Epoch%d.params'%(class_names[identifier], epoch))
            net.output.save_params('./model/%s_o_Epoch%d.params'%(class_names[identifier], epoch))

        if val_acc >= best_acc:
            best_acc = val_acc
            net.features.save_params('./model/%s_f2_Epoch%d.params'%(class_names[identifier], epoch))
            net.output.save_params('./model/%s_o2_Epoch%d.params'%(class_names[identifier], epoch))