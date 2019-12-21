import os
import argparse


def run(args):
    os.system(
        'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
            str(args.idx), 20, args.pool, args.dataset))
    os.system(
        'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
            str(args.idx), 32, args.pool, args.dataset))
    os.system(
        'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
            str(args.idx), 44, args.pool, args.dataset))
    os.system(
        'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
            str(args.idx), 56, args.pool, args.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--idx', default=1, help='')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--pool', default='FM', help='method to whiten')
    parser.add_argument('--layer', default=20, type=int, help='layer')
    parser.add_argument('--dataset', default='cifar10', help='norm of fm-out')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)
