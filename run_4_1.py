import os
import argparse


def run(args):
    os.system(
        'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
            114, 'dynamic', str(3), args.in_norm_fn, 'cifar10', True, True,
            100, str(args.gpu)
        ))
    # if args.gpu=='0':
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'svhn_cropped', False, False,
    #             50, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'fashion_mnist', False, False,
    #             50, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'cifar10', False, False,
    #             50, str(args.gpu)
    #         ))
    #
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'svhn_cropped', False, True,
    #             100, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'fashion_mnist', False, True,
    #             100, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(3), args.in_norm_fn, 'cifar10', True, True,
    #             100, str(args.gpu)
    #         ))
    #
    # if args.gpu=='1':
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'svhn_cropped', False, False,
    #             50, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'fashion_mnist', False, False,
    #             50, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'cifar10', False, False,
    #             50, str(args.gpu)
    #         ))
    #
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'svhn_cropped', False, True,
    #             100, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'fashion_mnist', False, True,
    #             100, str(args.gpu)
    #         ))
    #     os.system(
    #         'python models/ex4_1_light.py --idx={} --pool={} --iter_num={} --in_norm_fn={} --dataset={} --flip={} --crop={} --epochs={} --gpu={}'.format(
    #             str(args.idx), 'dynamic', str(2), args.in_norm_fn, 'cifar10', True, True,
    #             100, str(args.gpu)
    #         ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--idx', default=1, help='')
    parser.add_argument('--gpu', default='0', help='which gpu to use')
    parser.add_argument('--iter_num', default=3, type=int, help='iter_num')
    parser.add_argument('--routing', default='FM', help='method to pool')
    parser.add_argument('--in_norm_fn', default='squash', help='norm of in-caps')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--flip', default=True, help='flip')
    parser.add_argument('--crop', default=True, help='crop')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
    run(arguments)


