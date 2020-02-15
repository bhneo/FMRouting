import os
import argparse


def run(args):
    # resnet-cifar10 FM caps
    os.system(
        'python models/ex4_2.py --idx={} --pool=FM --dataset={} --atoms={} --gpu={} --flip={} --crop={} --epoch=163'.format(
            str(args.idx), 'fashion_mnist', str(args.atoms), str(args.gpu), args.flip, args.crop))
    os.system(
        'python models/ex4_2.py --idx={} --pool=FM --dataset={} --atoms={} --gpu={} --flip={} --crop={} --epoch=163'.format(
            str(args.idx), 'cifar10', str(args.atoms), str(args.gpu), args.flip, args.crop))
    os.system(
        'python models/ex4_2.py --idx={} --pool=FM --dataset={} --atoms={} --gpu={} --flip={} --crop={} --epoch=163'.format(
            str(args.idx), 'svhn_cropped', str(args.atoms), str(args.gpu), False, args.crop))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--idx', default=1, type=int, help='')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--flip', default=True, help='method to whiten')
    parser.add_argument('--crop', default=True, help='method to whiten')
    parser.add_argument('--atoms', default=16, type=int, help='atoms per capsule')
    parser.add_argument('--dataset', default='cifar10', help='norm of fm-out')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)
