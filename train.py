import argparse
import mxnet as mx
import os
import sys
from train_net import train_net, train_net_resp


def parse_args():
    parser = argparse.ArgumentParser(description='Train a chexnet network')
    parser.add_argument('--train-csv', dest='train_csv', help='.csv file to use',
                        default=os.path.join(os.getcwd(), 'data', 'Data_Entry.csv'), type=str)
    parser.add_argument('--network', dest='network', type=str, default='densenet121',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--epochs', dest='epochs', help='number of epochs of training',
                        default=100, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=224,
                        help='set image shape')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.03,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=list, default=[30, 60],
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=float, default=0.5,
                        help='ratio to refactor learning rate')
    parser.add_argument('--num-class', dest='num_class', type=int, default=14,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--identifier', dest='identifier', type=int, default=-1,
                        help='identifier(number) of the object of class to classify,for all if -1')

    args = parser.parse_args()
    return args

def parse_class_names(args):
    """ parse # classes and class_names if applicable """
    num_class = args.num_class
    if len(args.class_names) > 0:
        if os.path.isfile(args.class_names):
            # try to open it to read class names
            with open(args.class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in args.class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx
    # class names if applicable
    class_names = parse_class_names(args)
    # start training
    if (args.identifier > (args.num_class-1)) or (args.identifier < -1):
        print('Wrong identifier')
    elif args.identifier == -1:
        train_net(args.network, args.train_csv,
                  args.num_class, args.batch_size,
                  args.data_shape, ctx, args.epochs,
                  args.learning_rate, args.momentum, args.weight_decay,
                  args.lr_refactor_step, args.lr_refactor_ratio,
                  class_names=class_names,
                  optimizer=args.optimizer)
    else:
        train_net_resp(args.network, args.train_csv,
                       args.num_class, args.batch_size,
                       args.data_shape, ctx, args.epochs,
                       args.learning_rate, args.momentum, args.weight_decay,
                       args.lr_refactor_step, args.lr_refactor_ratio, args.identifier,
                       class_names=class_names,
                       optimizer=args.optimizer)