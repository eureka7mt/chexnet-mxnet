import argparse
import mxnet as mx
import os
import sys
from cam import Cam


def parse_args():
    parser = argparse.ArgumentParser(description='Class activation mapping demo')
    parser.add_argument('--network', dest='network', type=str, default='densenet121',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/00000377_004.png',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--model-path', dest='model_path', type=str,
                        default=os.path.join(os.getcwd(), 'model'),
                        help='trained model path')
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=224,
                        help='set image shape')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.4,
                        help='object visualize score threshold, default 0.4')
    parser.add_argument('--num-class', dest='num_class', type=int, default=14,
                        help='number of classes')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia',
                        help='string of comma separated names, or text filename')
    args = parser.parse_args()
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

if __name__ == '__main__':
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    image_list = [i.strip() for i in args.images.split(',')]
    assert len(image_list) > 0, "No valid image specified to detect"

    network = args.network
    class_names = parse_class_names(args.class_names)

    # run 
    Cam(network, image_list, args.model_path, ctx, args.data_shape, class_names, args.thresh,num_class=args.num_class)
