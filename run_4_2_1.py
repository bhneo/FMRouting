import os
import argparse


def run(args):
    if args.gpu == '0':
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 20, 'average', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 32, 'average', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 44, 'average', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 56, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(1), 110, 'average', args.dataset))

        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 20, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 32, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 44, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 56, 'average', args.dataset))
        #
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 20, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 32, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 44, 'average', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 56, 'average', args.dataset))
    else:
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 20, 'FM', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 32, 'FM', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 44, 'FM', args.dataset))
        os.system(
            'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
                str(1), 56, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(1), 110, 'FM', args.dataset))

        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 20, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 32, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 44, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(2), 56, 'FM', args.dataset))
        #
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 20, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 32, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 44, 'FM', args.dataset))
        # os.system(
        #     'python models/ex4_2_1.py --idx={} --layer_num={} --pool={} --dataset={} --flip=True --crop=True'.format(
        #         str(3), 56, 'FM', args.dataset))


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
